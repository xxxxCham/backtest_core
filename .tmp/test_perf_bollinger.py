"""Test performance bollinger_best_longe_3i"""
import numpy as np
from backtest.sweep_numba import _sweep_boll_level_long
import time

# Données test
n_bars = 10000
np.random.seed(42)
closes = (100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.02))).astype(np.float64)
highs = (closes * 1.01).astype(np.float64)
lows = (closes * 0.99).astype(np.float64)

# Grille params: TOUS les paramètres varient (cas réel)
n_combos = 5000
bb_periods = np.random.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100], n_combos).astype(np.float64)
bb_stds = np.random.uniform(1.0, 5.0, n_combos).astype(np.float64)
entry_levels = np.random.uniform(0.0, 0.5, n_combos).astype(np.float64)
sl_levels = np.random.uniform(-0.8, -0.1, n_combos).astype(np.float64)
tp_levels = np.random.uniform(0.5, 3.0, n_combos).astype(np.float64)
leverages = np.full(n_combos, 1.0, dtype=np.float64)
print(f'bb_period varie: {sorted(np.unique(bb_periods.astype(int)))}')

print(f'Test bollinger_best_longe_3i: {n_combos:,} combos × {n_bars:,} bars')
print('Warm-up JIT...')
_ = _sweep_boll_level_long(closes[:100], highs[:100], lows[:100],
    bb_periods[:5], bb_stds[:5], entry_levels[:5], sl_levels[:5], tp_levels[:5],
    leverages[:5], 10000.0, 10.0, 5.0)
print('✓ JIT compilé')

print(f'\n⚡ Exécution sweep...')
start = time.perf_counter()
pnls, sharpes, max_dds, win_rates, n_trades = _sweep_boll_level_long(
    closes, highs, lows, bb_periods, bb_stds, entry_levels, sl_levels, tp_levels,
    leverages, 10000.0, 10.0, 5.0)
elapsed = time.perf_counter() - start

print(f'\n{"="*60}')
print(f'⚡ Throughput: {n_combos/elapsed:,.0f} backtests/seconde')
print(f'⚡ Temps total: {elapsed:.3f}s')
print(f'⚡ Temps/bt: {elapsed/n_combos*1000:.3f} ms')
print(f'{"="*60}')
print(f'Best PnL: ${np.max(pnls):,.2f}')
print(f'Best Sharpe: {np.max(sharpes):.2f}')
