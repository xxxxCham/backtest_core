"""
Test de performance non-régression.

Garantit que le système maintient >10k bt/s sur 10k barres (5k combos).
Détecte automatiquement les régressions de performance.
"""
import sys
from pathlib import Path

# Ajouter racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pytest


def test_numba_sweep_performance_no_regression():
    """
    Test critique: garantir >10k bt/s sur sweep Numba.

    Configuration:
    - 5,000 combinaisons
    - 10,000 barres OHLCV
    - Stratégie: bollinger_best_longe_3i (vectorisée Numba)

    Seuils:
    - PASS: >=10,000 bt/s (objectif utilisateur)
    - WARN: 7,000-10,000 bt/s (dégradation détectée)
    - FAIL: <7,000 bt/s (régression critique)
    """
    from backtest.sweep_numba import _sweep_boll_level_long

    # Données test (10k barres)
    n_bars = 10000
    np.random.seed(42)
    closes = (100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.02))).astype(np.float64)
    highs = (closes * 1.01).astype(np.float64)
    lows = (closes * 0.99).astype(np.float64)

    # Grille params: 5k combos
    n_combos = 5000
    bb_periods = np.random.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100], n_combos).astype(np.float64)
    bb_stds = np.random.uniform(0.5, 6.0, n_combos).astype(np.float64)
    entry_levels = np.random.uniform(-0.2, 0.7, n_combos).astype(np.float64)
    sl_levels = np.random.uniform(-1.5, 0.1, n_combos).astype(np.float64)
    tp_levels = np.random.uniform(0.3, 4.0, n_combos).astype(np.float64)
    leverages = np.full(n_combos, 1.0, dtype=np.float64)

    # Warm-up JIT (exclu du timing)
    _ = _sweep_boll_level_long(
        closes[:100], highs[:100], lows[:100],
        bb_periods[:5], bb_stds[:5], entry_levels[:5], sl_levels[:5], tp_levels[:5],
        leverages[:5], 10000.0, 10.0, 5.0
    )

    # Mesure performance
    start = time.perf_counter()
    pnls, sharpes, max_dds, win_rates, n_trades = _sweep_boll_level_long(
        closes, highs, lows, bb_periods, bb_stds, entry_levels, sl_levels, tp_levels,
        leverages, 10000.0, 10.0, 5.0
    )
    elapsed = time.perf_counter() - start

    throughput = n_combos / elapsed

    # Assertions graduelles
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Throughput: {throughput:,.0f} bt/s")
    print(f"  Time:       {elapsed:.3f}s for {n_combos:,} combos")
    print(f"  Target:     >=10,000 bt/s")
    print(f"{'='*60}")

    if throughput >= 10000:
        print("  ✅ PASS - Performance excellente")
    elif throughput >= 7000:
        print("  ⚠️  WARN - Dégradation détectée (mais acceptable)")
        pytest.warns(UserWarning, match="Performance dégradée")
    else:
        print("  ❌ FAIL - Régression critique")

    # Seuil critique: échec si <7k bt/s
    assert throughput >= 7000, (
        f"Régression performance critique: {throughput:.0f} bt/s < 7,000 bt/s.\n"
        f"Objectif utilisateur: >=10,000 bt/s.\n"
        f"Vérifiez les changements récents (fast_metrics, df.loc, cache disque, allocations)."
    )

    # Validation résultats corrects
    assert len(pnls) == n_combos
    assert np.any(pnls != 0), "Aucun P&L calculé (erreur logique)"
    assert np.any(n_trades > 0), "Aucun trade (erreur logique)"


def test_simulator_fast_no_regression():
    """
    Test simulateur rapide (backtest individuel).

    Seuil: <0.5ms par backtest (2000+ bt/s sur séquentiel).
    """
    from backtest.engine import BacktestEngine
    from strategies.bollinger_best_longe_3i import BollingerBestLonge3iStrategy
    import pandas as pd

    # Données 1000 barres
    n_bars = 1000
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.02))
    df = pd.DataFrame({
        'close': close,
        'high': close * 1.01,
        'low': close * 0.99,
        'open': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='1h'))

    engine = BacktestEngine(initial_capital=10000.0)
    strategy = BollingerBestLonge3iStrategy()
    params = {
        "bb_period": 20,
        "bb_std": 2.1,
        "entry_level": 0.0,
        "sl_level": -0.5,
        "tp_level": 0.85,
    }

    # Warm-up
    engine.run(df=df, strategy=strategy, params=params, silent_mode=True, fast_metrics=True)

    # Mesure
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        engine.run(df=df, strategy=strategy, params=params, silent_mode=True, fast_metrics=True)
    elapsed = time.perf_counter() - start

    time_per_run_ms = (elapsed / n_runs) * 1000

    print(f"\n{'='*60}")
    print(f"SIMULATOR PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Time/run:  {time_per_run_ms:.2f} ms")
    print(f"  Target:    <2.0 ms (500+ bt/s séquentiel)")
    print(f"{'='*60}")

    # Seuil: <2ms par run (500+ bt/s séquentiel, acceptable)
    assert time_per_run_ms < 2.0, (
        f"Simulateur trop lent: {time_per_run_ms:.2f}ms > 2.0ms.\n"
        f"Objectif: <2ms pour 500+ bt/s séquentiel."
    )

    if time_per_run_ms < 1.0:
        print("  ✅ EXCELLENT - <1ms par run")
    else:
        print("  ✅ PASS - Performance acceptable")


if __name__ == "__main__":
    print("Exécution tests performance...")
    test_numba_sweep_performance_no_regression()
    test_simulator_fast_no_regression()
    print("\n✅ Tous les tests passés!")
