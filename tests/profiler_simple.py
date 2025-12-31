"""
Module-ID: profiler_simple

Purpose: Profiler simple BacktestEngine - exploiter PerfCounters natives (time_indicators, time_signals, etc.).

Role in pipeline: performance profiling

Key components: engine.run(), consultation result.counters, timing

Inputs: 1000 barres synthétiques, grille 4 paramètres

Outputs: PerfCounters (time_indicators, time_signals, time_trading, etc.)

Dependencies: numpy, pandas, backtest.engine, utils.config

Conventions: Compteurs en millisecondes

Read-if: Identifier bottlenecks via compteurs natifs engine.

Skip-if: Profiling pas besoin (ou utilisez profiler_detailed).
"""
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from backtest.engine import BacktestEngine
from utils.config import Config

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

# Accumuler les timings de chaque run
timings = defaultdict(list)

# Benchmark
print(f"\nLancement de {N_RUNS} backtests...\n")
start_time = time.perf_counter()

for i, params in enumerate(params_grid):
    engine = BacktestEngine(initial_capital=10000, config=config)

    result = engine.run(
        df=df,
        strategy='ema_cross',
        params=params,
        symbol='TEST',
        timeframe='5min',
        silent_mode=True  # Activer le mode silencieux pour tester les gains de performance
    )

    # Collecter les timings de ce run
    if 'perf_counters' in result.meta:
        counters = result.meta['perf_counters']

        # Extraire les durées
        if 'durations_ms' in counters:
            for key, value in counters['durations_ms'].items():
                timings[key + '_ms'].append(value)

        # Extraire les counts
        if 'counts' in counters:
            for key, value in counters['counts'].items():
                timings[key].append(value)

        # Extraire total_ms si présent
        if 'total_ms' in counters:
            timings['total_real_ms'].append(counters['total_ms'])

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N_RUNS} runs complétés...")

elapsed = time.perf_counter() - start_time

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

# Timings détaillés par étape
print(f"\n{'Étape':<25} {'Total (s)':<12} {'Moy (ms)':<12} {'% Total':<10}")
print('-' * 70)

# Calculer le temps total moyen
total_ms_list = timings.get('total_ms', [])
avg_total_ms = sum(total_ms_list) / len(total_ms_list) if total_ms_list else elapsed * 1000 / N_RUNS

# Trier les étapes par temps décroissant
steps = {}
for key in timings.keys():
    if key.endswith('_ms'):
        step_name = key.replace('_ms', '')
        times = timings[key]
        avg_time = sum(times) / len(times)
        total_time = sum(times) / 1000  # Convertir en secondes
        steps[step_name] = (total_time, avg_time)

sorted_steps = sorted(steps.items(), key=lambda x: x[1][0], reverse=True)

for step_name, (total_s, avg_ms) in sorted_steps:
    pct = (avg_ms / avg_total_ms) * 100 if avg_total_ms > 0 else 0
    print(f"{step_name:<25} {total_s:<12.3f} {avg_ms:<12.2f} {pct:<9.1f}%")

# Identifier les bottlenecks
print(f"\n{'=' * 70}")
print("TOP 3 BOTTLENECKS:")
print('=' * 70)

# Exclure 'total' du classement des bottlenecks
bottlenecks = [(name, total_s) for name, (total_s, avg_ms) in sorted_steps if name != 'total'][:3]

for i, (name, total_time) in enumerate(bottlenecks, 1):
    pct = (steps[name][1] / avg_total_ms) * 100
    print(f"{i}. {name:<25} {total_time:.3f}s ({pct:.1f}%)")

# Statistiques supplémentaires
if 'trades_count' in timings:
    total_trades = sum(timings['trades_count'])
    avg_trades = total_trades / N_RUNS
    print("\nStatistiques:")
    print(f"  Trades totaux: {total_trades}")
    print(f"  Trades moyens par run: {avg_trades:.1f}")

if 'signals_count' in timings:
    total_signals = sum(timings['signals_count'])
    avg_signals = total_signals / N_RUNS
    print(f"  Signaux totaux: {total_signals}")
    print(f"  Signaux moyens par run: {avg_signals:.1f}")

# Recommandations
print(f"\n{'=' * 70}")
print("RECOMMANDATIONS:")
print('=' * 70)

if runs_per_sec < 100:
    print(f"❌ Performance actuelle: {runs_per_sec:.1f} runs/sec (< 100)")
    print("   Optimisations urgentes nécessaires")
elif runs_per_sec < 300:
    print(f"⚠️ Performance actuelle: {runs_per_sec:.1f} runs/sec (< 300)")
    print("   Des optimisations pourraient aider")
else:
    print(f"✅ Bonne performance: {runs_per_sec:.0f} runs/sec")

if runs_per_sec < 500:
    print("\n💡 Objectif: 500 runs/sec")
    print(f"   Gap à combler: +{500 - runs_per_sec:.0f} runs/sec ({(500/runs_per_sec - 1)*100:.0f}% plus rapide)")

    # Suggérer optimisations basées sur les bottlenecks
    if bottlenecks:
        top_bottleneck = bottlenecks[0][0]
        top_pct = (steps[top_bottleneck][1] / avg_total_ms) * 100

        if top_pct > 40:
            print(f"\n🎯 Focus prioritaire: optimiser '{top_bottleneck}' ({top_pct:.0f}% du temps)")

            if top_bottleneck == 'metrics':
                print("   → Désactiver métriques non-essentielles")
                print("   → Vectoriser calculs (numpy)")
                print("   → Éviter copies de DataFrames")
            elif top_bottleneck == 'simulation':
                print("   → Utiliser simulator_fast (Numba) si disponible")
                print("   → Pré-extraire arrays numpy")
                print("   → Éviter df.loc[] dans boucles")
            elif top_bottleneck == 'indicators':
                print("   → Cacher indicateurs communs")
                print("   → Utiliser rolling windows numpy")
                print("   → Éviter recalculs redondants")
