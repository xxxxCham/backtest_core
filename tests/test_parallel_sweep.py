"""
Module-ID: test_parallel_sweep

Purpose: Test parallélisation SweepEngine - vérifier 8 workers exécution parallèle.

Role in pipeline: testing

Key components: SweepEngine, parallel workers, timing validation

Inputs: Grille paramétres, stratégie RSI

Outputs: Vérification temps exécution (parallèle < série)

Dependencies: pytest, backtest.sweep, time

Conventions: N workers = 8; mes timing d'exécution

Read-if: Vérifier parallélisation sweep fonctionne.

Skip-if: Performance parée connue.
"""

import time

import numpy as np
import pandas as pd

from backtest.sweep import SweepEngine
from strategies.rsi_reversal import RSIReversalStrategy  # Remplace RSITrendFilteredStrategy (inexistant)


def generate_test_data(n_bars: int = 1000) -> pd.DataFrame:
    """Génère des données OHLCV synthétiques."""
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='h')  # 'h' au lieu de '1H' (deprecated)

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 0.1,
        'high': close + np.abs(np.random.randn(n_bars) * 0.3),
        'low': close - np.abs(np.random.randn(n_bars) * 0.3),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)  # ✅ Utiliser dates comme index (DatetimeIndex)

    # Nommer l'index pour clarté
    df.index.name = 'timestamp'

    return df


def test_parallel_speedup():
    """Test du speedup avec 100 combinaisons."""

    print("=" * 70)
    print("TEST DE PARALLÉLISATION DU SWEEP ENGINE")
    print("=" * 70)

    # Générer données de test
    print("\nGénération des données de test (20000 bars)...")
    df = generate_test_data(n_bars=20000)
    strategy = RSIReversalStrategy()  # Remplace RSITrendFilteredStrategy

    # Grille de 100 combinaisons (10 x 10) - adaptée pour RSIReversalStrategy
    param_grid = {
        'rsi_period': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        'oversold': [25, 27, 29, 31, 33, 35, 37, 39, 41, 43],  # Renommé de rsi_oversold
    }

    print(f"Grille de paramètres: {len(param_grid['rsi_period'])} x {len(param_grid['oversold'])} = 100 combinaisons")

    # Test 1: Version séquentielle (1 worker)
    print("\n" + "=" * 70)
    print("[1/2] Test séquentiel (1 worker)...")
    print("=" * 70)
    engine_seq = SweepEngine(max_workers=1, use_processes=True, auto_save=False)
    start = time.time()
    results_seq = engine_seq.run_sweep(
        df=df,
        strategy=strategy,
        param_grid=param_grid,
        show_progress=True,
    )
    time_seq = time.time() - start

    print("\n✅ Séquentiel terminé:")
    print(f"   - Temps: {time_seq:.2f}s")
    print(f"   - Complétés: {results_seq.n_completed}/100")
    print(f"   - Échoués: {results_seq.n_failed}")
    print(f"   - Throughput: {results_seq.n_completed / time_seq:.2f} backtests/s")

    # Test 2: Version parallèle (8 workers)
    print("\n" + "=" * 70)
    print("[2/2] Test parallèle (8 workers)...")
    print("=" * 70)
    engine_par = SweepEngine(max_workers=8, use_processes=True, auto_save=False)
    start = time.time()
    results_par = engine_par.run_sweep(
        df=df,
        strategy=strategy,
        param_grid=param_grid,
        show_progress=True,
    )
    time_par = time.time() - start

    print("\n✅ Parallèle terminé:")
    print(f"   - Temps: {time_par:.2f}s")
    print(f"   - Complétés: {results_par.n_completed}/100")
    print(f"   - Échoués: {results_par.n_failed}")
    print(f"   - Throughput: {results_par.n_completed / time_par:.2f} backtests/s")

    # Calcul du speedup
    speedup = time_seq / time_par
    efficiency = (speedup / 8) * 100  # % d'efficacité parallèle

    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"Temps séquentiel:  {time_seq:.2f}s")
    print(f"Temps parallèle:   {time_par:.2f}s")
    print(f"Speedup:           {speedup:.2f}x")
    print(f"Efficacité:        {efficiency:.1f}% (idéal: 100% = 8x speedup)")
    print(f"Overhead:          {100 - efficiency:.1f}% (pickle, IPC, scheduling)")
    print("=" * 70)

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if speedup >= 4.0:
        print("\n✅ SUCCÈS: Speedup >= 4x")
        print("   La parallélisation fonctionne correctement.")
        print("   Les 8 workers s'exécutent bien en parallèle.")
        status = "PASS"
    elif speedup >= 2.0:
        print("\n⚠️  ATTENTION: Speedup entre 2-4x")
        print("   La parallélisation fonctionne mais est sous-optimale.")
        print("   Causes possibles:")
        print("   - Overhead de pickle/serialization élevé")
        print("   - Contention CPU/RAM")
        print("   - Backtests trop courts (< 1s)")
        status = "WARNING"
    else:
        print("\n❌ ÉCHEC: Speedup < 2x")
        print("   Problème de parallélisation détecté.")
        print("   Causes possibles:")
        print("   - Exécution séquentielle (chunking non corrigé)")
        print("   - GIL Python (devrait utiliser multiprocessing)")
        print("   - Workers qui s'attendent mutuellement")
        status = "FAIL"

    print("=" * 70)

    # Comparaison des résultats
    print("\n" + "=" * 70)
    print("COMPARAISON DES RÉSULTATS")
    print("=" * 70)

    # Vérifier que les meilleurs paramètres sont identiques
    if results_seq.best_params == results_par.best_params:
        print("✅ Meilleurs paramètres identiques (cohérence OK)")
        print(f"   Params: {results_seq.best_params}")
    else:
        print("⚠️  Meilleurs paramètres différents")
        print(f"   Séquentiel: {results_seq.best_params}")
        print(f"   Parallèle:  {results_par.best_params}")

    # Vérifier Sharpe ratio
    sharpe_seq = results_seq.best_metrics.get('sharpe_ratio', 0)
    sharpe_par = results_par.best_metrics.get('sharpe_ratio', 0)

    if isinstance(sharpe_seq, (int, float)) and isinstance(sharpe_par, (int, float)):
        sharpe_diff = abs(sharpe_seq - sharpe_par)
        if sharpe_diff < 0.01:
            print(f"✅ Sharpe ratio identique: {sharpe_seq:.4f}")
        else:
            print("⚠️  Sharpe ratio légèrement différent:")
            print(f"   Séquentiel: {sharpe_seq:.4f}")
            print(f"   Parallèle:  {sharpe_par:.4f}")
            print(f"   Diff: {sharpe_diff:.4f}")

    print("=" * 70)

    # Résumé final
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"Status:       {status}")
    print(f"Speedup:      {speedup:.2f}x")
    print(f"Efficacité:   {efficiency:.1f}%")
    print(f"Workers OK:   {'✅ Oui' if status == 'PASS' else '⚠️ Partiellement' if status == 'WARNING' else '❌ Non'}")
    print("=" * 70)

    return {
        'status': status,
        'time_seq': time_seq,
        'time_par': time_par,
        'speedup': speedup,
        'efficiency': efficiency,
        'n_completed_seq': results_seq.n_completed,
        'n_completed_par': results_par.n_completed,
    }


if __name__ == '__main__':
    try:
        results = test_parallel_speedup()

        # Exit code basé sur le status
        exit_code = 0 if results['status'] == 'PASS' else 1
        exit(exit_code)

    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        exit(2)
