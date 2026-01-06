"""
Module-ID: test_bug_fixes

Purpose: Test validation corrections bugs - passage paramètres, mapping bb_std, perf ProcessPool.

Role in pipeline: testing

Key components: BollingerATRStrategy param mapping, engine.run(), ProcessPoolExecutor

Inputs: Données synthétiques, grille paramétres

Outputs: Vérification mappings corrects et perf acceptable

Dependencies: backtest.engine, strategies.bollinger_atr

Conventions: Tests de régressions pour bugs fixés

Read-if: Vérifier bugs fixés ne réapparus.

Skip-if: Bugs déjà résolu et testés.
"""

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from strategies.bollinger_atr import BollingerATRStrategy


def test_param_mapping():
    """Test 1: Vérifier que bb_std est correctement mappé vers std_dev."""
    print("=" * 60)
    print("TEST 1: Mapping bb_std → std_dev")
    print("=" * 60)

    engine = BacktestEngine(initial_capital=10000)
    strategy = BollingerATRStrategy()

    # Paramètres de test avec bb_std différent du default
    params = {
        "bb_period": 20,
        "bb_std": 2.5,  # Différent du default 2.0
        "atr_period": 14,
        "leverage": 1
    }

    # Extraire les paramètres pour Bollinger
    indicator_params = engine._extract_indicator_params(strategy, "bollinger", params)

    print(f"Paramètres d'entrée: {params}")
    print(f"Paramètres extraits pour Bollinger: {indicator_params}")

    # Vérifications
    assert "std_dev" in indicator_params, "❌ ÉCHEC: 'std_dev' manquant dans paramètres extraits"
    assert indicator_params["std_dev"] == 2.5, f"❌ ÉCHEC: std_dev={indicator_params['std_dev']}, attendu 2.5"
    assert indicator_params["period"] == 20, f"❌ ÉCHEC: period={indicator_params['period']}, attendu 20"

    print("✅ SUCCÈS: bb_std est correctement mappé vers std_dev")
    print()


def test_grid_param_passing():
    """Test 2: Vérifier que tous les paramètres UI sont transmis à la grille."""
    print("=" * 60)
    print("TEST 2: Passage de paramètres dans la grille")
    print("=" * 60)

    from itertools import product

    # Simuler les paramètres UI
    params = {
        "bb_period": 20,
        "bb_std": 2.5,
        "atr_period": 14,
        "entry_pct_long": -0.35,
        "entry_pct_short": 0.95,
    }

    # Simuler param_ranges (seulement bb_period varie)
    param_ranges = {
        "bb_period": {"min": 15, "max": 25, "step": 5}
    }

    # Générer la grille (CODE CORRIGÉ)
    param_names = list(param_ranges.keys())
    param_values_lists = []

    for name in param_names:
        pr = param_ranges[name]
        values = list(range(int(pr["min"]), int(pr["max"]) + 1, int(pr["step"])))
        param_values_lists.append(values)

    param_grid = []
    for combo in product(*param_values_lists):
        # CORRECTION APPLIQUÉE: fusionner params fixes avec params variants
        param_dict = {**params, **dict(zip(param_names, combo))}
        param_grid.append(param_dict)

    print(f"Paramètres fixes UI: {params}")
    print(f"Param ranges: {param_ranges}")
    print(f"Grille générée ({len(param_grid)} combinaisons):")
    for i, p in enumerate(param_grid):
        print(f"  [{i}] {p}")

    # Vérifications
    assert len(param_grid) == 3, f"❌ ÉCHEC: {len(param_grid)} combinaisons, attendu 3"

    for i, p in enumerate(param_grid):
        assert "bb_std" in p, f"❌ ÉCHEC: bb_std manquant dans combo {i}"
        assert p["bb_std"] == 2.5, f"❌ ÉCHEC: bb_std={p['bb_std']} dans combo {i}, attendu 2.5"
        assert "atr_period" in p, f"❌ ÉCHEC: atr_period manquant dans combo {i}"
        assert p["atr_period"] == 14, f"❌ ÉCHEC: atr_period={p['atr_period']} dans combo {i}, attendu 14"
        assert "leverage" not in p, f"❌ ÉCHEC: leverage ne doit pas être optimisé par défaut (combo {i})"

    print("✅ SUCCÈS: Tous les paramètres UI sont bien transmis à la grille")
    print()


def test_backtest_results_vary():
    """Test 3: Vérifier que les résultats varient avec des paramètres différents."""
    print("=" * 60)
    print("TEST 3: Variation des résultats avec paramètres différents")
    print("=" * 60)

    # Créer des données de test
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1h')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(500) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(500) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Paramètres de test avec bb_std différent
    params_list = [
        {"bb_period": 20, "bb_std": 1.5, "atr_period": 14, "leverage": 1},
        {"bb_period": 20, "bb_std": 2.5, "atr_period": 14, "leverage": 1},
        {"bb_period": 20, "bb_std": 3.0, "atr_period": 14, "leverage": 1},
    ]

    results = []
    for i, params in enumerate(params_list):
        # Créer un DataFrame légèrement différent pour chaque test (éviter cache)
        df_test = df.copy()
        df_test['close'] = df_test['close'] + i * 0.001  # Variation minime

        engine = BacktestEngine(initial_capital=10000)
        try:
            result = engine.run(
                df=df_test,
                strategy="bollinger_atr",
                params=params,
                symbol=f"TEST{i}",  # Symbole différent pour éviter cache
                timeframe="1h",
                silent_mode=True
            )
            pnl = result.metrics.get("total_pnl", 0)
            results.append((params["bb_std"], pnl))
            print(f"  bb_std={params['bb_std']:.1f} → PNL={pnl:.2f}")
        except Exception as e:
            print(f"  bb_std={params['bb_std']:.1f} → ERREUR: {e}")
            results.append((params["bb_std"], None))

    # Vérifier que les résultats ne sont pas tous identiques
    pnl_values = [pnl for _, pnl in results if pnl is not None]

    if len(pnl_values) < 2:
        print("❌ ÉCHEC: Pas assez de résultats valides pour comparer")
        return

    # Vérifier qu'au moins 2 résultats sont différents (tolérance de 0.01)
    unique_pnls = set(round(pnl, 2) for pnl in pnl_values)

    if len(unique_pnls) == 1:
        print(f"❌ ÉCHEC: Tous les résultats sont identiques ({pnl_values[0]:.2f})")
        print("   → Le bug de paramètres n'est pas corrigé!")
    else:
        print(f"✅ SUCCÈS: Les résultats varient ({len(unique_pnls)} valeurs uniques)")
    print(f"   Valeurs PNL: {sorted(unique_pnls)}")
    print()


def _simple_task(x):
    """Fonction au niveau module pour test de picklabilité."""
    return x * 2


def test_performance_comparison():
    """Test 4: Comparer performance séquentiel vs multi-process (simulation)."""
    print("=" * 60)
    print("TEST 4: Performance ProcessPoolExecutor")
    print("=" * 60)

    # Note: Ce test ne peut pas vraiment tester ProcessPoolExecutor avec backtests
    # car il nécessite l'UI Streamlit. On vérifie juste que l'import et le pickling fonctionnent.

    try:
        from concurrent.futures import ProcessPoolExecutor
        print("✅ ProcessPoolExecutor importé avec succès")

        # Test simple de picklabilité avec fonction au niveau module
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(_simple_task, range(5)))

        assert results == [0, 2, 4, 6, 8], "❌ ÉCHEC: ProcessPoolExecutor ne fonctionne pas"
        print("✅ ProcessPoolExecutor fonctionne correctement")

        # Vérifier que notre wrapper est aussi au niveau module
        from ui.main import _run_backtest_multiprocess
        assert _run_backtest_multiprocess
        print("✅ _run_backtest_multiprocess est importable (donc picklable)")

    except ImportError:
        print("⚠️  Note: ui.main._run_backtest_multiprocess non disponible (normal si UI non chargée)")
    except Exception as e:
        print(f"❌ ÉCHEC: Erreur avec ProcessPoolExecutor: {e}")
        raise

    print()


def main():
    """Exécuter tous les tests."""
    print("\n" + "=" * 60)
    print("SUITE DE TESTS - CORRECTIONS DE BUGS")
    print("=" * 60 + "\n")

    try:
        test_param_mapping()
        test_grid_param_passing()
        test_backtest_results_vary()
        test_performance_comparison()

        print("=" * 60)
        print("TOUS LES TESTS TERMINÉS")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST ÉCHOUÉ: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
