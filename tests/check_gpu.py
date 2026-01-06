#!/usr/bin/env python3
"""
Script de diagnostic GPU pour backtest_core
Vérifie CuPy, CUDA, et l'utilisation GPU pendant les backtests
"""

import os
import sys

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_cupy():
    """Vérifier si CuPy est disponible"""
    try:
        import cupy as cp
        print("✅ CuPy installé")
        print(f"   Version: {cp.__version__}")

        # Tester allocation GPU
        test_array = cp.array([1, 2, 3])
        print(f"   Test allocation GPU: {test_array.device}")

        return True
    except ImportError as e:
        print("❌ CuPy non installé")
        print(f"   Erreur: {e}")
        return False
    except Exception as e:
        print("❌ CuPy installé mais erreur CUDA")
        print(f"   Erreur: {e}")
        return False


def check_gpu_backend():
    """Vérifier le backend GPU de backtest_core"""
    try:
        from performance.gpu import HAS_CUPY, gpu_available, get_gpu_info

        print("\n📊 Backend GPU backtest_core:")
        print(f"   HAS_CUPY: {HAS_CUPY}")
        print(f"   GPU disponible: {gpu_available()}")

        if gpu_available():
            info = get_gpu_info()
            print("   GPU Info:")
            for key, val in info.items():
                print(f"      {key}: {val}")

        return HAS_CUPY
    except Exception as e:
        print(f"\n❌ Erreur import performance.gpu: {e}")
        return False


def check_indicators_gpu():
    """Vérifier si les indicateurs utilisent le GPU"""
    try:
        from indicators.registry import _INDICATORS
        import inspect

        print("\n🔍 Analyse des indicateurs:")

        gpu_enabled = 0
        for name, func in _INDICATORS.items():
            source = inspect.getsource(func)
            if 'cupy' in source.lower() or 'cp.' in source or 'device_backend' in source:
                print(f"   ✅ {name}: GPU activé")
                gpu_enabled += 1
            else:
                print(f"   ⏸️ {name}: NumPy uniquement")

        print(f"\n   Total: {gpu_enabled}/{len(_INDICATORS)} avec support GPU")

        return gpu_enabled > 0
    except Exception as e:
        print(f"\n❌ Erreur analyse indicateurs: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_engine_gpu():
    """Vérifier si le moteur de backtest utilise le GPU"""
    try:
        from backtest.engine import BacktestEngine
        import inspect

        print("\n🔧 Analyse du moteur de backtest:")
        source = inspect.getsource(BacktestEngine)

        has_cupy = 'cupy' in source.lower() or 'device_backend' in source.lower()

        if has_cupy:
            print("   ✅ Intégration GPU trouvée dans BacktestEngine")
        else:
            print("   ❌ Aucune intégration GPU dans BacktestEngine")

        return has_cupy
    except Exception as e:
        print(f"\n❌ Erreur analyse BacktestEngine: {e}")
        return False


def main():
    """Exécuter tous les diagnostics"""
    print("=" * 60)
    print("🔍 DIAGNOSTIC GPU - backtest_core")
    print("=" * 60)

    # 1. CuPy disponible ?
    cupy_ok = check_cupy()

    # 2. Backend GPU configuré ?
    backend_ok = check_gpu_backend()

    # 3. Indicateurs utilisent GPU ?
    indicators_ok = check_indicators_gpu()

    # 4. Engine utilise GPU ?
    engine_ok = check_engine_gpu()

    # Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ")
    print("=" * 60)

    issues = []

    if not cupy_ok:
        issues.append("❌ CuPy non fonctionnel (installer via requirements-gpu.txt)")
    else:
        print("✅ CuPy fonctionnel")

    if not backend_ok:
        issues.append("❌ Backend GPU non activé")
    else:
        print("✅ Backend GPU activé")

    if not indicators_ok:
        issues.append("⚠️ Aucun indicateur n'utilise le GPU")
    else:
        print("✅ Au moins un indicateur utilise le GPU")

    if not engine_ok:
        issues.append("❌ BacktestEngine n'utilise pas le GPU")
    else:
        print("✅ BacktestEngine intègre le GPU")

    if issues:
        print("\n🔴 PROBLÈMES IDENTIFIÉS:")
        for issue in issues:
            print(f"   {issue}")

        print("\n💡 SOLUTION:")
        if not cupy_ok:
            print("   1. Installer CuPy: pip install -r requirements-gpu.txt")
        if not indicators_ok:
            print("   2. Modifier les indicateurs pour utiliser device_backend")
        if not engine_ok:
            print("   3. Intégrer device_backend dans BacktestEngine.run()")
    else:
        print("\n✅ Tout est OK !")

    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
