"""
Test réaliste des optimisations Numba dans un contexte de sweep.

Mesure les performances sur appels multiples (après warm-up JIT).
"""
import time
import numpy as np
import pandas as pd

from backtest.performance_numba import (
    _expanding_max_numba,
    _drawdown_series_numba,
    _ulcer_index_numba,
)


def benchmark_realistic():
    """
    Benchmark réaliste: 30 backtests comme dans test_performance.py
    """
    print("="*70)
    print("BENCHMARK RÉALISTE - 30 Backtests")
    print("="*70)

    # Générer 30 equity curves (simulant 30 backtests)
    np.random.seed(42)
    equity_curves = []
    for i in range(30):
        equity = np.cumsum(np.random.randn(116654) * 10) + 10000
        equity[equity < 0] = 100
        equity_curves.append(equity)

    print(f"✓ 30 equity curves générées (116,654 barres chacune)")

    # ===================================================================
    # Test 1: drawdown_series (appelé dans calculate_metrics)
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 1: drawdown_series × 30")
    print("="*70)

    # Warm-up Numba (compilation JIT)
    _ = _drawdown_series_numba(equity_curves[0])

    # Benchmark Pandas
    t0 = time.perf_counter()
    for equity in equity_curves:
        s = pd.Series(equity)
        running_max = s.expanding().max()
        dd = (s / running_max) - 1.0
    t1 = time.perf_counter()
    time_pandas = t1 - t0

    # Benchmark Numba
    t0 = time.perf_counter()
    for equity in equity_curves:
        dd = _drawdown_series_numba(equity)
    t1 = time.perf_counter()
    time_numba = t1 - t0

    speedup_dd = time_pandas / time_numba
    print(f"Pandas: {time_pandas:.3f}s ({time_pandas*1000/30:.1f}ms/backtest)")
    print(f"Numba:  {time_numba:.3f}s ({time_numba*1000/30:.1f}ms/backtest)")
    print(f"Speedup: {speedup_dd:.1f}×")

    # ===================================================================
    # Test 2: ulcer_index (appelé dans Tier S metrics)
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 2: ulcer_index × 30")
    print("="*70)

    # Warm-up Numba
    _ = _ulcer_index_numba(equity_curves[0])

    # Benchmark Pandas
    t0 = time.perf_counter()
    for equity in equity_curves:
        s = pd.Series(equity)
        running_max = s.expanding().max()
        drawdown_pct = ((s / running_max) - 1.0) * 100
        squared_dd = drawdown_pct ** 2
        ulcer = np.sqrt(squared_dd.mean())
    t1 = time.perf_counter()
    time_pandas = t1 - t0

    # Benchmark Numba
    t0 = time.perf_counter()
    for equity in equity_curves:
        ulcer = _ulcer_index_numba(equity)
    t1 = time.perf_counter()
    time_numba = t1 - t0

    speedup_ui = time_pandas / time_numba
    print(f"Pandas: {time_pandas:.3f}s ({time_pandas*1000/30:.1f}ms/backtest)")
    print(f"Numba:  {time_numba:.3f}s ({time_numba*1000/30:.1f}ms/backtest)")
    print(f"Speedup: {speedup_ui:.1f}×")

    # ===================================================================
    # Test 3: expanding_max (utilisé partout)
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 3: expanding_max × 30")
    print("="*70)

    # Warm-up Numba
    _ = _expanding_max_numba(equity_curves[0])

    # Benchmark Pandas
    t0 = time.perf_counter()
    for equity in equity_curves:
        s = pd.Series(equity)
        running_max = s.expanding().max()
    t1 = time.perf_counter()
    time_pandas = t1 - t0

    # Benchmark Numba
    t0 = time.perf_counter()
    for equity in equity_curves:
        running_max = _expanding_max_numba(equity)
    t1 = time.perf_counter()
    time_numba = t1 - t0

    speedup_em = time_pandas / time_numba
    print(f"Pandas: {time_pandas:.3f}s ({time_pandas*1000/30:.1f}ms/backtest)")
    print(f"Numba:  {time_numba:.3f}s ({time_numba*1000/30:.1f}ms/backtest)")
    print(f"Speedup: {speedup_em:.1f}×")

    # ===================================================================
    # Résumé
    # ===================================================================
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"drawdown_series:  {speedup_dd:>5.1f}× speedup")
    print(f"ulcer_index:      {speedup_ui:>5.1f}× speedup")
    print(f"expanding_max:    {speedup_em:>5.1f}× speedup")

    avg_speedup = (speedup_dd + speedup_ui + speedup_em) / 3
    print(f"\nMoyenne:          {avg_speedup:>5.1f}× speedup")

    if avg_speedup > 1.0:
        print(f"\n✅ Numba est {avg_speedup:.1f}× plus rapide que Pandas!")
        print("Les optimisations sont bénéfiques pour les sweeps.")
    else:
        print(f"\n⚠️ Numba est {1/avg_speedup:.1f}× plus lent que Pandas!")
        print("Les optimisations ne sont pas bénéfiques.")

    print("="*70)


if __name__ == "__main__":
    benchmark_realistic()
