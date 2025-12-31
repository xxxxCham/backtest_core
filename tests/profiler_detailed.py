"""
Module-ID: profiler_detailed

Purpose: Profiler détaillé BacktestEngine pour identifier bottlenecks - timing par étape (indicateurs, signaux, trading).

Role in pipeline: performance profiling

Key components: DetailedProfiler, engine.run(), granular timing

Inputs: 1000 barres synthétiques, grille 3 paramètres

Outputs: Timings détaillés par fonction, analyse bottlenecks

Dependencies: numpy, pandas, time, backtest.engine

Conventions: Profiling manuel par fonction; temps en millisecondes

Read-if: Deep dive bottlenecks, optimisation spécifiques.

Skip-if: Baseline perf suffisant ou utilisez cProfile.
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from backtest.config import Config
from backtest.engine import BacktestEngine

# Générer données OHLCV synthétiques
np.random.seed(42)
n_bars = 1000
dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')

prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
df = pd.DataFrame({
    'open': prices + np.random.randn(n_bars) * 0.1,
    'high': prices + abs(np.random.randn(n_bars) * 0.3),
    'low': prices - abs(np.random.randn(n_bars) * 0.3),
    'close': prices,
    'volume': np.random.randint(1000, 10000, n_bars)
}, index=dates)

# Configuration de test
config = Config()
params_grid = [
    {'k_entry': 0.5, 'k_tp': 1.0, 'k_sl': 0.5, 'trailing_stop': False},
    {'k_entry': 0.8, 'k_tp': 1.5, 'k_sl': 0.8, 'trailing_stop': False},
    {'k_entry': 1.0, 'k_tp': 2.0, 'k_sl': 1.0, 'trailing_stop': True},
    {'k_entry': 0.6, 'k_tp': 1.2, 'k_sl': 0.6, 'trailing_stop': False},
    {'k_entry': 0.7, 'k_tp': 1.8, 'k_sl': 0.7, 'trailing_stop': True},
] * 20  # 100 runs

N_RUNS = len(params_grid)

print(f"Profiler BacktestEngine - {N_RUNS} runs sur {n_bars} barres")
print("=" * 70)

# Dictionnaire pour stocker les temps
timings = defaultdict(list)

# Sauvegarder la méthode originale
original_run = BacktestEngine.run


def profiled_run(self, df, strategy, params=None, **kwargs):
    """Version instrumentée de run() qui mesure chaque étape"""
    t0 = time.perf_counter()

    # 1. Préparation
    t_start = time.perf_counter()
    from backtest.strategies import get_strategy
    strat_obj = get_strategy(strategy) if isinstance(strategy, str) else strategy
    params = params or {}
    timings['1_preparation'].append(time.perf_counter() - t_start)

    # 2. Indicateurs
    t_start = time.perf_counter()
    df_with_indicators = strat_obj.compute_indicators(df)
    timings['2_indicators'].append(time.perf_counter() - t_start)

    # 3. Signaux
    t_start = time.perf_counter()
    signals = strat_obj.generate_signals(df_with_indicators, **params)
    timings['3_signals'].append(time.perf_counter() - t_start)

    # 4. Validation
    t_start = time.perf_counter()
    if signals.empty or signals['signal'].abs().sum() == 0:
        timings['4_validation'].append(time.perf_counter() - t_start)
        return {'trades': [], 'metrics': {}}
    timings['4_validation'].append(time.perf_counter() - t_start)

    # 5. Exécution (simulation)
    t_start = time.perf_counter()
    from backtest.execution import execute_trades
    trades, equity = execute_trades(
        df=df_with_indicators,
        signals=signals,
        initial_capital=self.initial_capital,
        params=params,
        config=self.config
    )
    timings['5_execution'].append(time.perf_counter() - t_start)

    # 6. Métriques
    t_start = time.perf_counter()
    from backtest.performance import compute_metrics
    metrics = compute_metrics(
        trades=trades,
        equity=equity,
        df=df_with_indicators,
        initial_capital=self.initial_capital,
        run_id=self.run_id
    )
    timings['6_metrics'].append(time.perf_counter() - t_start)

    # 7. Packaging résultats
    t_start = time.perf_counter()
    result = {
        'trades': trades,
        'equity': equity,
        'metrics': metrics,
        'signals': signals
    }
    timings['7_packaging'].append(time.perf_counter() - t_start)

    timings['TOTAL'].append(time.perf_counter() - t0)

    return result


# Remplacer temporairement
BacktestEngine.run = profiled_run

# Benchmark
print(f"Lancement de {N_RUNS} backtests...\n")
start_time = time.perf_counter()

for i, params in enumerate(params_grid):
    engine = BacktestEngine(initial_capital=10000, config=config)
    result = engine.run(
        df=df,
        strategy='crossing',
        params=params
    )

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N_RUNS} runs completés...")

elapsed = time.perf_counter() - start_time

# Restaurer la méthode originale
BacktestEngine.run = original_run

# Analyse des résultats
print(f"\n{'=' * 70}")
print("RÉSULTATS DU PROFILING")
print('=' * 70)

runs_per_sec = N_RUNS / elapsed if elapsed > 0 else 0
bars_per_sec = (N_RUNS * n_bars) / elapsed if elapsed > 0 else 0

print("\nPerformance globale:")
print(f"  Temps total: {elapsed:.2f}s")
print(f"  Runs/sec: {runs_per_sec:.1f}")
print(f"  Barres/sec: {bars_per_sec:.0f}")
print(f"  Temps moyen par run: {elapsed/N_RUNS*1000:.1f}ms")

print(f"\n{'Étape':<20} {'Total (s)':<12} {'Moy (ms)':<12} {'% Total':<10} {'Appels/s'}")
print('-' * 70)

total_time = sum(timings['TOTAL'])

for key in sorted(timings.keys()):
    times = timings[key]
    total = sum(times)
    mean_ms = (total / len(times)) * 1000
    pct = (total / total_time) * 100 if total_time > 0 else 0
    calls_per_sec = len(times) / total if total > 0 else 0

    print(f"{key:<20} {total:<12.3f} {mean_ms:<12.2f} {pct:<9.1f}% {calls_per_sec:>8.0f}")

# Identifier les bottlenecks
print(f"\n{'=' * 70}")
print("TOP 3 BOTTLENECKS:")
print('=' * 70)

bottlenecks = [(k, sum(v)) for k, v in timings.items() if k != 'TOTAL']
bottlenecks.sort(key=lambda x: x[1], reverse=True)

for i, (name, total_time) in enumerate(bottlenecks[:3], 1):
    pct = (total_time / sum(timings['TOTAL'])) * 100
    print(f"{i}. {name:<20} {total_time:.3f}s ({pct:.1f}%)")

# Recommandations
print(f"\n{'=' * 70}")
print("RECOMMANDATIONS:")
print('=' * 70)

if runs_per_sec < 100:
    print("❌ Performance < 100 runs/sec")
    if sum(timings.get('6_metrics', [0])) / total_time > 0.3:
        print("   → Optimiser le calcul des métriques (>30% du temps)")
    if sum(timings.get('5_execution', [0])) / total_time > 0.4:
        print("   → Optimiser la simulation (>40% du temps)")
    if sum(timings.get('2_indicators', [0])) / total_time > 0.2:
        print("   → Optimiser le calcul des indicateurs (>20% du temps)")
elif runs_per_sec < 300:
    print("⚠️ Performance < 300 runs/sec - Optimisations possibles")
else:
    print(f"✅ Bonne performance: {runs_per_sec:.0f} runs/sec")

if runs_per_sec < 500:
    gain_possible = 500 - runs_per_sec
    print(f"\n💡 Objectif: 500 runs/sec (gain possible: +{gain_possible:.0f} runs/sec)")
