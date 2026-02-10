"""Test impact taille chunks sur saturation CPU."""
import os
import sys
import time
sys.path.insert(0, '.')

import numpy as np
from backtest.sweep_numba import _sweep_boll_level_long

# Données 10k barres
n_bars = 10000
np.random.seed(42)
closes = (100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.02))).astype(np.float64)
highs = (closes * 1.01).astype(np.float64)
lows = (closes * 0.99).astype(np.float64)

# Grille 20k combos (pour tester distribution)
n_combos = 20000
print(f"Test avec {n_combos:,} combos × {n_bars:,} bars")
print(f"Workers disponibles: ~24-32")
print(f"Chunk size actuel: {os.getenv('NUMBA_CHUNK_SIZE', '50000 (défaut)')}\n")

bb_periods = np.random.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100], n_combos).astype(np.float64)
bb_stds = np.random.uniform(0.5, 6.0, n_combos).astype(np.float64)
entry_levels = np.random.uniform(-0.2, 0.7, n_combos).astype(np.float64)
sl_levels = np.random.uniform(-1.5, 0.1, n_combos).astype(np.float64)
tp_levels = np.random.uniform(0.3, 4.0, n_combos).astype(np.float64)
leverages = np.full(n_combos, 1.0, dtype=np.float64)

# Warm-up
_ = _sweep_boll_level_long(
    closes[:100], highs[:100], lows[:100],
    bb_periods[:5], bb_stds[:5], entry_levels[:5], sl_levels[:5], tp_levels[:5],
    leverages[:5], 10000.0, 10.0, 5.0
)

# Test performance
print("Exécution sweep (surveillez CPU dans Gestionnaire des tâches)...")
start = time.perf_counter()
pnls, sharpes, max_dds, win_rates, n_trades = _sweep_boll_level_long(
    closes, highs, lows, bb_periods, bb_stds, entry_levels, sl_levels, tp_levels,
    leverages, 10000.0, 10.0, 5.0
)
elapsed = time.perf_counter() - start

throughput = n_combos / elapsed

print(f"\n{'='*60}")
print(f"RÉSULTATS")
print(f"{'='*60}")
print(f"  Throughput: {throughput:,.0f} bt/s")
print(f"  Temps:      {elapsed:.2f}s")
print(f"  Efficacité: {(throughput/100000)*100:.1f}% (vs 100k bt/s théorique)")
print(f"{'='*60}")
print("\nSi CPU en YOYO:")
print("  → Essayez: set NUMBA_CHUNK_SIZE=10000")
print("  → Ou:      set NUMBA_CHUNK_SIZE=5000")
print("  → Objectif: CPU stable à ~90-100%")
