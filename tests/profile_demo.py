"""
Profiler Demo - Test avec données synthétiques

Génère des données OHLCV synthétiques pour tester le profiler
sans avoir besoin de vraies données de marché.

Usage:
    python tools/profile_demo.py
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
    strategies_module = importlib.import_module("strategies")
    profiler_module = importlib.import_module("tools.profiler")
    config_module = importlib.import_module("utils.config")

    return (
        profiler_module.PerformanceProfiler,
        engine_module.BacktestEngine,
        config_module.Config,
        strategies_module.get_strategy,
    )


def generate_synthetic_ohlcv(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "1h",
    initial_price: float = 50000.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """
    Génère des données OHLCV synthétiques.

    Args:
        start_date: Date de début
        end_date: Date de fin
        timeframe: Timeframe
        initial_price: Prix initial
        volatility: Volatilité (0.02 = 2%)

    Returns:
        DataFrame OHLCV avec index DatetimeIndex
    """
    # Calculer le nombre de barres
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    freq_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
    freq = freq_map.get(timeframe, "1H")

    # Générer les timestamps
    timestamps = pd.date_range(start=start, end=end, freq=freq)
    n_bars = len(timestamps)

    # Générer les prix (random walk)
    np.random.seed(42)  # Pour reproductibilité
    returns = np.random.normal(0, volatility, n_bars)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Générer OHLC
    open_prices = prices
    high_prices = prices * (1 + np.abs(np.random.normal(0, volatility / 2, n_bars)))
    low_prices = prices * (1 - np.abs(np.random.normal(0, volatility / 2, n_bars)))
    close_prices = prices * (1 + np.random.normal(0, volatility / 3, n_bars))

    # Volume
    volumes = np.random.uniform(100, 1000, n_bars)

    # Créer le DataFrame
    df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    }, index=timestamps)

    return df


def demo_simple_backtest():
    """Démo : Profiler un backtest simple."""
    performance_profiler_cls, backtest_engine_cls, config_cls, get_strategy_fn = _bootstrap()
    print("\n" + "=" * 80)
    print("DEMO 1 : PROFILING BACKTEST SIMPLE")
    print("=" * 80)

    # Générer des données synthétiques
    print("\n📊 Génération de données synthétiques...")
    df = generate_synthetic_ohlcv(
        start_date="2024-01-01",
        end_date="2024-12-31",
        timeframe="1h",
    )
    print(f"   ✅ {len(df)} barres générées\n")
    print(f"   DEBUG: df.shape={df.shape}, df.index type={type(df.index)}")
    print(f"   DEBUG: first={df.index[0] if len(df) > 0 else 'N/A'}")

    # Paramètres de la stratégie
    strategy_name = "ema_cross"
    strategy_class = get_strategy_fn(strategy_name)
    strategy_instance = strategy_class()  # Instancier la stratégie
    params = {
        spec.name: spec.default
        for spec in strategy_instance.parameter_specs.values()
    }

    # Créer le moteur
    config = config_cls()
    engine = backtest_engine_cls(initial_capital=10000.0, config=config)

    # Profiler
    profiler = performance_profiler_cls()
    profiler.start()

    result = engine.run(df, strategy_name, params)

    profiler.stop()

    # Résultats
    if result:
        print(f"\n{'=' * 80}")
        print("RÉSULTATS BACKTEST")
        print(f"{'=' * 80}")
        print(f"PnL Total: ${result.metrics['total_pnl']:.2f}")
        print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {result.metrics['win_rate']:.2f}%")
        print(f"Total Trades: {result.metrics['total_trades']}")
        print(f"{'=' * 80}\n")

    # Stats profiling
    profiler.print_stats(top_n=20, sort_by="cumulative")

    # Sauvegarder
    profiler.save_report("demo_simple")

    return profiler


def demo_grid_search():
    """Démo : Profiler une optimisation Grid Search."""
    performance_profiler_cls, backtest_engine_cls, config_cls, _ = _bootstrap()
    print("\n" + "=" * 80)
    print("DEMO 2 : PROFILING GRID SEARCH")
    print("=" * 80)

    # Générer des données synthétiques
    print("\n📊 Génération de données synthétiques...")
    df = generate_synthetic_ohlcv(
        start_date="2024-01-01",
        end_date="2024-06-30",
        timeframe="4h",
    )
    print(f"   ✅ {len(df)} barres générées\n")

    # Paramètres à tester
    strategy_name = "ema_cross"
    param_grid = [
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 12, "slow_period": 26},
        {"fast_period": 15, "slow_period": 35},
        {"fast_period": 20, "slow_period": 50},
    ]

    print(f"🔢 {len(param_grid)} combinaisons à tester\n")

    # Créer le moteur
    config = config_cls()
    engine = backtest_engine_cls(initial_capital=10000.0, config=config)

    # Profiler
    profiler = performance_profiler_cls()
    profiler.start()

    results = []
    for i, params in enumerate(param_grid):
        result = engine.run(df, strategy_name, params)
        if result:
            results.append({
                "params": params,
                "sharpe": result.metrics["sharpe_ratio"],
                "pnl": result.metrics["total_pnl"],
            })
        print(f"   Progression: {i+1}/{len(param_grid)} ({(i+1)/len(param_grid)*100:.0f}%)")

    profiler.stop()

    # Meilleurs résultats
    if results:
        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df["sharpe"].idxmax()]

        print(f"\n{'=' * 80}")
        print("MEILLEURS RÉSULTATS")
        print(f"{'=' * 80}")
        print(f"Meilleur Sharpe: {best['sharpe']:.3f}")
        print(f"PnL: ${best['pnl']:.2f}")
        print(f"Paramètres: {best['params']}")
        print(f"{'=' * 80}\n")

    # Stats profiling
    profiler.print_stats(top_n=20, sort_by="cumulative")

    # Sauvegarder
    profiler.save_report("demo_grid")

    return profiler


def main():
    """Point d'entrée principal."""
    print("\n" + "🔍 " * 20)
    print("PROFILER DEMO - Backtest Core")
    print("🔍 " * 20)

    # Demo 1 : Backtest simple
    demo_simple_backtest()

    # Demo 2 : Grid Search
    demo_grid_search()

    # Instructions
    print("\n" + "=" * 80)
    print("📋 PROCHAINES ÉTAPES")
    print("=" * 80)
    print("\n1. Les rapports .prof ont été sauvegardés dans profiling_results/")
    print("\n2. Pour générer un rapport HTML :")
    print("   python tools/profile_analyzer.py --report profiling_results/demo_simple_*.prof --output demo_analysis.html")
    print("\n3. Ouvrir le rapport :")
    print("   start demo_analysis.html")
    print("\n4. Chercher les fonctions en ROUGE (> 10% du temps) pour optimiser")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
