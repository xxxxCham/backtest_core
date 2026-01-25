"""
Test de validation des optimisations Numba.

Vérifie que les résultats sont identiques aux versions Pandas.
Mesure le gain de performance réel.
"""
import time
import numpy as np
import pandas as pd

from backtest.performance_numba import (
    _expanding_max_numba,
    _drawdown_series_numba,
    _max_drawdown_numba,
    _ulcer_index_numba,
    _recovery_factor_numba,
)


def test_expanding_max():
    """Test _expanding_max_numba vs pandas.expanding().max()"""
    print("\n" + "="*70)
    print("TEST: expanding_max")
    print("="*70)

    # Générer données de test
    np.random.seed(42)
    data = np.cumsum(np.random.randn(100000)) + 1000

    # Version Pandas
    t0 = time.perf_counter()
    s = pd.Series(data)
    result_pandas = s.expanding().max().values
    t1 = time.perf_counter()
    time_pandas = (t1 - t0) * 1000

    # Version Numba
    t0 = time.perf_counter()
    result_numba = _expanding_max_numba(data)
    t1 = time.perf_counter()
    time_numba = (t1 - t0) * 1000

    # Validation
    diff = np.abs(result_pandas - result_numba).max()
    speedup = time_pandas / time_numba

    print(f"Temps Pandas: {time_pandas:.2f}ms")
    print(f"Temps Numba:  {time_numba:.2f}ms")
    print(f"Speedup:      {speedup:.1f}×")
    print(f"Max diff:     {diff:.10f}")

    if diff < 1e-10:
        print("✅ VALIDATION OK - Résultats identiques")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE - Différence détectée")
        return False


def test_drawdown_series():
    """Test _drawdown_series_numba vs pandas"""
    print("\n" + "="*70)
    print("TEST: drawdown_series")
    print("="*70)

    # Générer equity curve
    np.random.seed(42)
    equity = np.cumsum(np.random.randn(100000) * 10) + 10000
    equity[equity < 0] = 100  # Éviter valeurs négatives

    # Version Pandas
    t0 = time.perf_counter()
    s = pd.Series(equity)
    running_max = s.expanding().max()
    result_pandas = ((s / running_max) - 1.0).values
    t1 = time.perf_counter()
    time_pandas = (t1 - t0) * 1000

    # Version Numba
    t0 = time.perf_counter()
    result_numba = _drawdown_series_numba(equity)
    t1 = time.perf_counter()
    time_numba = (t1 - t0) * 1000

    # Validation
    diff = np.abs(result_pandas - result_numba).max()
    speedup = time_pandas / time_numba

    print(f"Temps Pandas: {time_pandas:.2f}ms")
    print(f"Temps Numba:  {time_numba:.2f}ms")
    print(f"Speedup:      {speedup:.1f}×")
    print(f"Max diff:     {diff:.10f}")

    if diff < 1e-10:
        print("✅ VALIDATION OK - Résultats identiques")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE - Différence détectée")
        return False


def test_max_drawdown():
    """Test _max_drawdown_numba"""
    print("\n" + "="*70)
    print("TEST: max_drawdown")
    print("="*70)

    # Générer equity curve
    np.random.seed(42)
    equity = np.cumsum(np.random.randn(100000) * 10) + 10000
    equity[equity < 0] = 100

    # Version Pandas
    t0 = time.perf_counter()
    s = pd.Series(equity)
    running_max = s.expanding().max()
    dd = (s / running_max) - 1.0
    result_pandas = dd.min()
    t1 = time.perf_counter()
    time_pandas = (t1 - t0) * 1000

    # Version Numba
    t0 = time.perf_counter()
    result_numba = _max_drawdown_numba(equity)
    t1 = time.perf_counter()
    time_numba = (t1 - t0) * 1000

    # Validation
    diff = abs(result_pandas - result_numba)
    speedup = time_pandas / time_numba

    print(f"Temps Pandas: {time_pandas:.2f}ms")
    print(f"Temps Numba:  {time_numba:.2f}ms")
    print(f"Speedup:      {speedup:.1f}×")
    print(f"Diff:         {diff:.10f}")

    if diff < 1e-10:
        print("✅ VALIDATION OK - Résultats identiques")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE - Différence détectée")
        return False


def test_ulcer_index():
    """Test _ulcer_index_numba"""
    print("\n" + "="*70)
    print("TEST: ulcer_index")
    print("="*70)

    # Générer equity curve
    np.random.seed(42)
    equity = np.cumsum(np.random.randn(100000) * 10) + 10000
    equity[equity < 0] = 100

    # Version Pandas
    t0 = time.perf_counter()
    s = pd.Series(equity)
    running_max = s.expanding().max()
    drawdown_pct = ((s / running_max) - 1.0) * 100
    squared_dd = drawdown_pct ** 2
    result_pandas = np.sqrt(squared_dd.mean())
    t1 = time.perf_counter()
    time_pandas = (t1 - t0) * 1000

    # Version Numba
    t0 = time.perf_counter()
    result_numba = _ulcer_index_numba(equity)
    t1 = time.perf_counter()
    time_numba = (t1 - t0) * 1000

    # Validation
    diff = abs(result_pandas - result_numba)
    speedup = time_pandas / time_numba

    print(f"Temps Pandas: {time_pandas:.2f}ms")
    print(f"Temps Numba:  {time_numba:.2f}ms")
    print(f"Speedup:      {speedup:.1f}×")
    print(f"Diff:         {diff:.10f}")

    if diff < 1e-6:  # Tolérance légèrement plus large pour sqrt
        print("✅ VALIDATION OK - Résultats identiques")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE - Différence détectée")
        return False


def test_recovery_factor():
    """Test _recovery_factor_numba"""
    print("\n" + "="*70)
    print("TEST: recovery_factor")
    print("="*70)

    # Générer equity curve
    np.random.seed(42)
    equity = np.cumsum(np.random.randn(100000) * 10) + 10000
    equity[equity < 0] = 100
    initial_capital = 10000.0

    # Version Pandas
    t0 = time.perf_counter()
    s = pd.Series(equity)
    net_profit = s.iloc[-1] - initial_capital
    running_max = s.expanding().max()
    drawdown_abs = running_max - s
    max_dd_abs = drawdown_abs.max()
    result_pandas = net_profit / max_dd_abs if max_dd_abs > 1e-10 else 0.0
    t1 = time.perf_counter()
    time_pandas = (t1 - t0) * 1000

    # Version Numba
    t0 = time.perf_counter()
    result_numba = _recovery_factor_numba(equity, initial_capital)
    t1 = time.perf_counter()
    time_numba = (t1 - t0) * 1000

    # Validation
    diff = abs(result_pandas - result_numba)
    speedup = time_pandas / time_numba

    print(f"Temps Pandas: {time_pandas:.2f}ms")
    print(f"Temps Numba:  {time_numba:.2f}ms")
    print(f"Speedup:      {speedup:.1f}×")
    print(f"Diff:         {diff:.10f}")

    if diff < 1e-6:
        print("✅ VALIDATION OK - Résultats identiques")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE - Différence détectée")
        return False


def main():
    """Exécuter tous les tests"""
    print("="*70)
    print("VALIDATION OPTIMISATIONS NUMBA")
    print("="*70)
    print("Barres de test: 100,000 (équivalent BTCUSDC/30m)")

    results = []
    results.append(("expanding_max", test_expanding_max()))
    results.append(("drawdown_series", test_drawdown_series()))
    results.append(("max_drawdown", test_max_drawdown()))
    results.append(("ulcer_index", test_ulcer_index()))
    results.append(("recovery_factor", test_recovery_factor()))

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)

    all_passed = all(result for _, result in results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    print("="*70)
    if all_passed:
        print("✅ TOUS LES TESTS RÉUSSIS")
        print("Les optimisations Numba sont validées et prêtes!")
        return 0
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifier les différences ci-dessus")
        return 1


if __name__ == "__main__":
    exit(main())
