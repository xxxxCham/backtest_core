"""
Diagnostic des Métriques - Analyse Approfondie
==============================================

Vérifie la cohérence des calculs de métriques et identifie les bugs potentiels.
"""

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache
def _bootstrap():
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    engine_module = importlib.import_module("backtest.engine")
    performance_module = importlib.import_module("backtest.performance")
    strategies_module = importlib.import_module("strategies")
    config_module = importlib.import_module("utils.config")

    return (
        engine_module.BacktestEngine,
        config_module.Config,
        performance_module.drawdown_series,
        performance_module.max_drawdown,
        strategies_module.get_strategy,
    )


def print_section(title: str):
    """Affiche une section avec bordure."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")


def test_drawdown_formula():
    """Teste la formule de drawdown avec différents scénarios."""
    _, _, drawdown_series_fn, max_drawdown_fn, _ = _bootstrap()
    print_section("🔍 TEST 1 : Formule de Drawdown")

    scenarios = [
        ("Capital positif stable", [10000, 10500, 10200, 11000, 10800]),
        ("Perte < 50%", [10000, 9000, 8500, 7000, 7500]),
        ("Perte 50-90%", [10000, 5000, 3000, 1500, 2000]),
        ("Perte > 100% (ruine)", [10000, 5000, 2000, -500, -1000]),
    ]

    for name, equity_values in scenarios:
        print(f"\n📊 Scénario : {name}")
        equity = pd.Series(equity_values, index=range(len(equity_values)))

        # Calcul drawdown
        dd = drawdown_series_fn(equity)
        max_dd = max_drawdown_fn(equity)

        print(f"  Équité : {equity.tolist()}")
        print(f"  Drawdown série : {[f'{x:.2%}' for x in dd.tolist()]}")
        print(f"  Max Drawdown : {max_dd:.2%}")

        # Vérifier si aberrant
        if max_dd < -1.0:  # Drawdown > 100%
            print(f"  ⚠️  ABERRANT : Drawdown impossible ({max_dd:.2%})")

            # Calcul alternatif sûr
            running_max = equity.expanding().max()
            # Clamp equity négative à 0 pour le calcul
            equity_safe = equity.clip(lower=0)
            dd_safe = (equity_safe / running_max) - 1.0
            max_dd_safe = dd_safe.min()

            print(f"  ✅ Correction proposée : {max_dd_safe:.2%}")


def test_real_backtest():
    """Teste avec un vrai backtest."""
    backtest_engine_cls, config_cls, _, _, get_strategy_fn = _bootstrap()
    print_section("🔍 TEST 2 : Backtest Réel")

    # Générer données synthétiques simples
    print("📊 Génération de données OHLCV...")
    np.random.seed(42)

    n_bars = 1000
    timestamps = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Random walk avec drift négatif (pour forcer des pertes)
    returns = np.random.normal(-0.001, 0.02, n_bars)
    prices = 50000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(100, 1000, n_bars),
    }, index=timestamps)

    print(f"  ✅ {len(df)} barres générées")
    print(f"  Prix : ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")

    # Backtest
    print("\n🔄 Exécution backtest EMA Cross...")
    strategy_class = get_strategy_fn("ema_cross")
    strategy = strategy_class()
    params = {
        spec.name: spec.default
        for spec in strategy.parameter_specs.values()
    }

    config = config_cls()
    engine = backtest_engine_cls(initial_capital=10000.0, config=config)
    result = engine.run(df, "ema_cross", params, silent_mode=True)

    if not result:
        print("❌ Backtest échoué")
        return

    # Analyser les métriques
    print("\n📈 Analyse des Métriques :")
    metrics = result.metrics

    print("\n  💰 Rendement :")
    print(f"    - PnL Total : ${metrics['total_pnl']:.2f}")
    print(f"    - Return % : {metrics['total_return_pct']:.2f}%")
    print(f"    - CAGR : {metrics['cagr']:.2f}%")

    print("\n  📉 Risque :")
    print(f"    - Sharpe : {metrics['sharpe_ratio']:.3f}")
    print(f"    - Sortino : {metrics['sortino_ratio']:.3f}")
    print(f"    - Max DD : {metrics['max_drawdown']:.2f}%")
    print(f"    - Volatilité : {metrics['volatility_annual']:.2f}%")

    print("\n  🎯 Trades :")
    print(f"    - Total : {metrics['total_trades']}")
    print(f"    - Win Rate : {metrics['win_rate']:.2f}%")
    print(f"    - Profit Factor : {metrics['profit_factor']:.2f}")

    # VÉRIFICATIONS
    print("\n🔎 Vérifications de Cohérence :")

    issues = []

    # 1. Drawdown impossible
    if metrics['max_drawdown'] < -100:
        issues.append(
            f"❌ Drawdown impossible : {metrics['max_drawdown']:.2f}% "
            "(ne peut pas dépasser -100%)"
        )

    # 2. Équité finale cohérente
    equity_final = result.equity.iloc[-1]
    expected_final = 10000 + metrics['total_pnl']
    if abs(equity_final - expected_final) > 0.01:
        issues.append(
            f"❌ Équité finale incohérente : {equity_final:.2f} "
            f"vs attendu {expected_final:.2f}"
        )

    # 3. Return % cohérent avec PnL
    calculated_return = (metrics['total_pnl'] / 10000) * 100
    if abs(metrics['total_return_pct'] - calculated_return) > 0.01:
        issues.append(
            f"❌ Return % incohérent : {metrics['total_return_pct']:.2f}% "
            f"vs calculé {calculated_return:.2f}%"
        )

    # 4. Win rate cohérent
    if result.trades is not None and len(result.trades) > 0:
        winning = (result.trades['pnl'] > 0).sum()
        calculated_wr = (winning / len(result.trades)) * 100
        if abs(metrics['win_rate'] - calculated_wr) > 0.01:
            issues.append(
                f"❌ Win Rate incohérent : {metrics['win_rate']:.2f}% "
                f"vs calculé {calculated_wr:.2f}%"
            )

    # 5. Équité négative
    if (result.equity < 0).any():
        min_equity = result.equity.min()
        issues.append(
            f"⚠️  Équité négative détectée : min = ${min_equity:.2f} "
            "(ruine du compte)"
        )

    if issues:
        print("\n🚨 PROBLÈMES DÉTECTÉS :")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ Toutes les vérifications passées")

    # Afficher courbe d'équité si problématique
    if metrics['max_drawdown'] < -100 or (result.equity < 0).any():
        print("\n📊 Courbe d'Équité (premiers/derniers points) :")
        print(f"  Début : {result.equity.head(10).tolist()}")
        print(f"  Fin : {result.equity.tail(10).tolist()}")
        print(f"  Min : ${result.equity.min():.2f}")
        print(f"  Max : ${result.equity.max():.2f}")


def test_drawdown_fix():
    """Teste une correction pour le calcul du drawdown."""
    _, _, _, max_drawdown_fn, _ = _bootstrap()
    print_section("🔧 TEST 3 : Correction Proposée pour Drawdown")

    # Cas problématique : équité négative
    equity = pd.Series([10000, 8000, 5000, 2000, -500, -1000])

    print("📊 Équité avec ruine :")
    print(f"  {equity.tolist()}")

    # Méthode actuelle (bugguée)
    max_dd_old = max_drawdown_fn(equity)

    print("\n❌ Méthode actuelle :")
    print(f"  Max Drawdown : {max_dd_old:.2%}")

    # Méthode corrigée : clamper à -100%
    running_max = equity.expanding().max()

    # Option 1 : Clamper l'équité négative à 0
    equity_clamped = equity.clip(lower=0)
    dd_fixed1 = (equity_clamped / running_max) - 1.0
    max_dd_fixed1 = dd_fixed1.min()

    # Option 2 : Calculer différence absolue puis ratio
    dd_abs = equity - running_max  # Perte absolue
    dd_fixed2 = (dd_abs / running_max).clip(lower=-1.0)  # Clamper à -100%
    max_dd_fixed2 = dd_fixed2.min()

    print("\n✅ Correction Option 1 (clamp equity à 0) :")
    print(f"  Max Drawdown : {max_dd_fixed1:.2%}")

    print("\n✅ Correction Option 2 (clamp drawdown à -100%) :")
    print(f"  Max Drawdown : {max_dd_fixed2:.2%}")

    print("\n💡 Recommandation : Option 2 (plus fidèle à la réalité)")


def main():
    """Point d'entrée principal."""
    print("\n" + "🔬 " * 20)
    print("DIAGNOSTIC MÉTRIQUES - Backtest Core")
    print("🔬 " * 20)

    # Tests
    test_drawdown_formula()
    test_real_backtest()
    test_drawdown_fix()

    # Conclusion
    print_section("📋 CONCLUSION")
    print("🔴 Bugs Identifiés :")
    print("  1. drawdown_series() ne gère pas les équités négatives")
    print("     → Donne des valeurs > -100% (mathématiquement impossibles)")
    print()
    print("  2. Pas de protection contre la ruine du compte")
    print("     → L'équité peut devenir négative (dette théorique)")
    print()
    print("✅ Correctifs Recommandés :")
    print("  1. Clamper le drawdown à -100% max")
    print("  2. Arrêter le backtest si équité <= 0 (ruine)")
    print("  3. Ajouter un flag 'account_ruined' dans les métriques")
    print()
    print("📖 Voir : docs/METRICS_FIX.md pour implémentation détaillée")
    print()


if __name__ == "__main__":
    main()
