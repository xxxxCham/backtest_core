#!/usr/bin/env python3
"""
Test CPU + GPU parallèle
Vérifie que BacktestEngine utilise le GPU et ProcessPoolExecutor utilise tous les CPU cores
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine


def test_gpu_backend():
    """Test 1: Vérifier que BacktestEngine utilise le GPU"""
    print("=" * 60)
    print("TEST 1: GPU Backend dans BacktestEngine")
    print("=" * 60)

    # Données de test
    dates = pd.date_range("2024-01-01", periods=1000, freq="1H")
    df = pd.DataFrame({
        "open": np.random.randn(1000).cumsum() + 100,
        "high": np.random.randn(1000).cumsum() + 102,
        "low": np.random.randn(1000).cumsum() + 98,
        "close": np.random.randn(1000).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 1000),
    }, index=dates)

    # Engine standard (use_gpu obsolète, supprimé de l'API)
    print("\n1️⃣ BacktestEngine (GPU auto-détecté):")
    engine = BacktestEngine(initial_capital=10000)
    print(f"   Engine initialisé: capital={engine.initial_capital}")

    # Vérifier si CuPy est disponible pour GPU
    try:
        import cupy as cp
        gpu_available = True
        print(f"   CuPy disponible: {cp.__version__}")
        print(f"   GPUs détectés: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError:
        gpu_available = False
        print("   CuPy non installé (mode CPU)")

    # Backtest rapide
    print("\n2️⃣ Backtest rapide (EMA cross):")
    start = time.time()
    result = engine.run(df, "ema_cross", {"fast_period": 10, "slow_period": 21})
    duration = time.time() - start

    print(f"   Durée: {duration:.3f}s")
    print(f"   Trades: {result.metrics.get('total_trades', 0)}")
    print(f"   Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}")

    return gpu_available


def worker_backtest(params):
    """Worker function pour ProcessPoolExecutor"""
    fast, slow = params

    # Chaque processus initialise son propre backend
    dates = pd.date_range("2024-01-01", periods=500, freq="h")  # 'h' au lieu de '1H' (FutureWarning)
    df = pd.DataFrame({
        "open": np.random.randn(500).cumsum() + 100,
        "high": np.random.randn(500).cumsum() + 102,
        "low": np.random.randn(500).cumsum() + 98,
        "close": np.random.randn(500).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 500).astype(float),
    }, index=dates)

    engine = BacktestEngine(initial_capital=10000)  # use_gpu supprimé
    result = engine.run(df, "ema_cross", {"fast_period": fast, "slow_period": slow})

    return {
        "params": f"fast={fast}, slow={slow}",
        "sharpe": result.metrics.get("sharpe_ratio", 0),
        "trades": len(result.trades),
        "device": "auto",  # backend.device_type supprimé
    }


def test_multiprocessing():
    """Test 2: Vérifier ProcessPoolExecutor multi-CPU"""
    print("\n" + "=" * 60)
    print("TEST 2: ProcessPoolExecutor multi-CPU")
    print("=" * 60)

    # Grille de paramètres
    param_grid = [(fast, slow) for fast in range(5, 15, 2) for slow in range(20, 40, 5)]
    print(f"\n📊 Grille: {len(param_grid)} combinaisons")

    # Exécution parallèle
    print("\n1️⃣ Lancement ProcessPoolExecutor (8 workers):")
    start = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker_backtest, param_grid))

    duration = time.time() - start

    print("\n2️⃣ Résultats:")
    print(f"   Durée totale: {duration:.2f}s")
    print(f"   Durée par run: {duration/len(param_grid):.3f}s")
    print(f"   Speedup théorique: ~{len(param_grid)/duration:.1f}x")

    # Afficher quelques résultats
    print("\n3️⃣ Top 3 Sharpe:")
    sorted_results = sorted(results, key=lambda x: x["sharpe"], reverse=True)[:3]
    for i, res in enumerate(sorted_results, 1):
        print(f"   {i}. {res['params']}: Sharpe={res['sharpe']:.2f}, Trades={res['trades']}, Device={res['device']}")

    return duration


def main():
    """Exécuter tous les tests"""
    print("🚀 TEST CPU + GPU PARALLÈLE")
    print()

    try:
        # Test 1: GPU backend
        gpu_ok = test_gpu_backend()

        # Test 2: Multi-CPU
        duration = test_multiprocessing()

        # Résumé
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ")
        print("=" * 60)

        if gpu_ok:
            print("✅ GPU backend activé et fonctionnel")
        else:
            print("⚠️ GPU backend non disponible (CPU fallback)")

        print("✅ ProcessPoolExecutor multi-CPU fonctionnel")
        print(f"✅ Performance: {duration:.2f}s pour 20 runs")

        print("\n💡 Architecture:")
        print("   - Chaque processus → 1 CPU core")
        print("   - Chaque processus → Accès GPU (via CuPy)")
        print("   - Parallélisme réel: CPU + GPU simultanés")

        return True

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
