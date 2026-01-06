"""
Profiling Spécifique - Métriques de Performance
==============================================

Profile uniquement le calcul des métriques pour identifier les bottlenecks.
"""

import cProfile
import importlib
import pstats
import sys
from datetime import datetime
from functools import lru_cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache
def _load_calculate_metrics():
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    module = importlib.import_module("backtest.performance")
    return getattr(module, "calculate_metrics")


def profile_metrics_calculation():
    """Profile le calcul des métriques."""
    calculate_metrics = _load_calculate_metrics()

    # Générer des données de test
    np.random.seed(42)
    n_points = 10000  # 10k points

    # Courbe d'équité réaliste
    initial_capital = 10000.0
    returns = np.random.normal(0.0001, 0.02, n_points)
    equity_values = initial_capital * np.exp(np.cumsum(returns))

    timestamps = pd.date_range(start="2024-01-01", periods=n_points, freq="h")
    equity = pd.Series(equity_values, index=timestamps)
    returns_series = pd.Series(returns, index=timestamps)

    # Trades DataFrame
    n_trades = 500
    trades_data = {
        "pnl": np.random.normal(10, 50, n_trades),
        "entry_ts": timestamps[:n_trades],
        "exit_ts": timestamps[100:n_trades+100],
    }
    trades_df = pd.DataFrame(trades_data)

    print("\n📊 Données de test :")
    print(f"  • Équité : {len(equity)} points")
    print(f"  • Returns : {len(returns_series)} points")
    print(f"  • Trades : {len(trades_df)}")
    print("\n⏱️  Profiling en cours...\n")

    # Profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Répéter 100 fois pour avoir des stats significatives
    for _ in range(100):
        metrics = calculate_metrics(
            equity=equity,
            returns=returns_series,
            trades_df=trades_df,
            initial_capital=initial_capital,
            include_tier_s=False,
            sharpe_method="daily_resample"
        )

    profiler.disable()

    # Analyser
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)  # Top 30

    print(stream.getvalue())

    # Sauvegarder
    output_path = Path("profiling_results")
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_file = output_path / f"metrics_{timestamp}.prof"
    stats.dump_stats(str(prof_file))

    print(f"\n📁 Rapport sauvegardé : {prof_file}")

    # Analyser les résultats
    print("\n📈 Exemple de métriques calculées :")
    print(f"  • Sharpe : {metrics['sharpe_ratio']:.3f}")
    print(f"  • Max DD : {metrics['max_drawdown']:.2f}%")
    print(f"  • Ruine : {metrics.get('account_ruined', False)}")
    print(f"  • Total Trades : {metrics['total_trades']}")


if __name__ == "__main__":
    print("\n🔬 PROFILING MÉTRIQUES - Focus Performance")
    print("=" * 60)
    profile_metrics_calculation()
    print("\n✅ Profiling terminé !")
