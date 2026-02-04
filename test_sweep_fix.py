"""
Script de test pour valider la correction du bug de sauvegarde des sweeps.

Vérifie :
1. Exécution du sweep
2. Sauvegarde automatique dans backtest_results/sweep_*/
3. Accès correct aux métriques via .items (pas .results)
4. Métriques accessibles comme dict (pas comme objet)
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.sweep import SweepEngine
from strategies.base import StrategyBase


class SimpleTestStrategy(StrategyBase):
    """Stratégie simple pour tester le sweep."""

    def init_defaults(self):
        return {
            "period": 20,
            "threshold": 0.5,
        }

    def required_indicators(self) -> list:
        """Pas d'indicateurs requis pour ce test."""
        return []

    def generate_signals(self, df: pd.DataFrame) -> tuple:
        """Génère des signaux aléatoires pour le test."""
        n = len(df)
        longs = np.random.random(n) > 0.7
        shorts = np.random.random(n) > 0.7
        stop_loss = np.full(n, 0.02)
        take_profit = np.full(n, 0.04)
        return longs, shorts, stop_loss, take_profit


def create_test_data(n_bars=1000):
    """Crée des données OHLCV de test."""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')

    # Prix aléatoires avec une tendance
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n_bars) * 0.1,
        'high': close + np.abs(np.random.randn(n_bars)) * 0.3,
        'low': close - np.abs(np.random.randn(n_bars)) * 0.3,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars),
    })

    df.set_index('timestamp', inplace=True)
    return df


def test_sweep_save():
    """Test principal du sweep avec sauvegarde."""
    print("=" * 60)
    print("TEST: Sweep avec sauvegarde automatique")
    print("=" * 60)

    # 1. Créer les données de test
    print("\n[1/5] Création des données de test...")
    df = create_test_data(n_bars=500)
    print(f"✓ {len(df)} barres créées")

    # 2. Créer la stratégie
    print("\n[2/5] Initialisation de la stratégie...")
    strategy = SimpleTestStrategy()
    print(f"✓ Stratégie: {strategy.__class__.__name__}")

    # 3. Définir une grille de paramètres PETITE (pour test rapide)
    param_grid = {
        "period": [10, 20, 30],
        "threshold": [0.3, 0.5, 0.7],
    }
    total_combos = 3 * 3
    print(f"\n[3/5] Grille de paramètres: {total_combos} combinaisons")
    print(f"  - period: {param_grid['period']}")
    print(f"  - threshold: {param_grid['threshold']}")

    # 4. Exécuter le sweep avec auto_save=True
    print(f"\n[4/5] Exécution du sweep...")
    engine = SweepEngine(
        max_workers=4,
        initial_capital=10000,
        auto_save=True  # ← CRITIQUE: Activer la sauvegarde auto
    )

    sweep_results = engine.run_sweep(
        df=df,
        strategy=strategy,
        param_grid=param_grid,
        optimize_for="sharpe_ratio",
        silent_mode=True,
        fast_metrics=True,
    )

    print(f"✓ Sweep terminé: {sweep_results.n_completed}/{total_combos} réussis")

    # 5. Vérifier les résultats
    print(f"\n[5/5] Vérification des résultats...")

    # 5a. Vérifier l'attribut .items (pas .results)
    try:
        items = sweep_results.items
        print(f"✓ Accès à sweep_results.items: {len(items)} éléments")
    except AttributeError as e:
        print(f"✗ ERREUR: Impossible d'accéder à .items: {e}")
        return False

    # 5b. Vérifier l'accès aux métriques comme dict
    if items:
        first_item = items[0]
        try:
            # Test accès dict (correct)
            sharpe = first_item.metrics.get("sharpe_ratio", 0)
            total_pnl = first_item.metrics.get("total_pnl", 0)
            print(f"✓ Accès dict aux métriques: sharpe={sharpe:.2f}, pnl={total_pnl:.2f}")
        except Exception as e:
            print(f"✗ ERREUR accès métriques: {e}")
            return False

    # 5c. Vérifier que le fichier de sauvegarde existe
    sweep_dirs = list(Path("backtest_results").glob("sweep_*"))
    if sweep_dirs:
        latest_sweep = max(sweep_dirs, key=os.path.getmtime)
        summary_file = latest_sweep / "summary.json"
        results_file = latest_sweep / "all_results.parquet"

        if summary_file.exists():
            print(f"✓ Fichier summary.json trouvé: {summary_file}")
        else:
            print(f"✗ ERREUR: summary.json manquant dans {latest_sweep}")
            return False

        if results_file.exists():
            print(f"✓ Fichier all_results.parquet trouvé: {results_file}")

            # Charger et vérifier le contenu
            df_results = pd.read_parquet(results_file)
            print(f"✓ Parquet chargé: {len(df_results)} lignes, {len(df_results.columns)} colonnes")

            # Vérifier les colonnes essentielles
            expected_cols = ["period", "threshold", "sharpe_ratio", "total_pnl", "success"]
            missing = [c for c in expected_cols if c not in df_results.columns]
            if missing:
                print(f"⚠ Colonnes manquantes: {missing}")
            else:
                print(f"✓ Toutes les colonnes essentielles présentes")

            print(f"\nAperçu des résultats:")
            print(df_results[["period", "threshold", "sharpe_ratio", "total_pnl"]].head())

        else:
            print(f"✗ ERREUR: all_results.parquet manquant dans {latest_sweep}")
            return False
    else:
        print(f"✗ ERREUR: Aucun répertoire sweep_* trouvé dans backtest_results/")
        return False

    # 6. Afficher les meilleurs paramètres
    print(f"\n" + "=" * 60)
    print("MEILLEURS PARAMÈTRES:")
    print("=" * 60)
    print(f"Params: {sweep_results.best_params}")
    print(f"Sharpe: {sweep_results.best_metrics.get('sharpe_ratio', 'N/A')}")
    print(f"Total PnL: {sweep_results.best_metrics.get('total_pnl', 'N/A')}")
    print(f"Win Rate: {sweep_results.best_metrics.get('win_rate_pct', 'N/A')}%")

    print(f"\n{'=' * 60}")
    print("✅ TOUS LES TESTS RÉUSSIS")
    print(f"{'=' * 60}\n")

    return True


if __name__ == "__main__":
    success = test_sweep_save()
    sys.exit(0 if success else 1)
