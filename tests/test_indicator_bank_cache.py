"""
Module-ID: tests.test_indicator_bank_cache

Purpose: Tester cache IndicatorBank (hit/miss, CPU vs GPU distinction, clés cache).

Role in pipeline: testing

Key components: test_* functions, create_sample_data fixture

Inputs: IndicatorBank, OHLCV DataFrame, backend (CPU/GPU), params

Outputs: Cache keys, hit_rate, persisted cache files

Dependencies: pytest, pandas, numpy, data.indicator_bank

Conventions: Cache distingue CPU/GPU par clé; hit_rate tracking; file-based persistence.

Read-if: Modification cache key generation ou backend distinction.

Skip-if: Tests cache non critiques.
"""

import pandas as pd
import numpy as np
import pytest
from data.indicator_bank import IndicatorBank


def create_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """Crée des données OHLCV de test."""
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='h')

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 0.1,
        'high': close + np.abs(np.random.randn(n_bars) * 0.3),
        'low': close - np.abs(np.random.randn(n_bars) * 0.3),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)

    df.index.name = 'timestamp'
    return df


class TestIndicatorBankCache:
    """Tests du système de cache avec backends CPU/GPU."""

    def setup_method(self):
        """Nettoyage avant chaque test."""
        bank = IndicatorBank()
        bank.clear()  # Vider le cache persistant

    def test_cache_key_differs_by_backend(self):
        """
        Test : Deux appels avec même data mais backends différents
        doivent produire des clés de cache différentes.
        """
        bank = IndicatorBank()
        df = create_sample_data(n_bars=1000)

        # Générer clés avec backends explicites
        key_cpu = bank._generate_key("rsi", {"period": 14, "_backend": "cpu"}, df)
        key_gpu = bank._generate_key("rsi", {"period": 14, "_backend": "gpu"}, df)

        # Les clés complètes doivent être différentes
        assert key_cpu[0] != key_gpu[0], "Clés doivent être différentes CPU vs GPU"

        # Les hash de params doivent être différents
        assert key_cpu[1] != key_gpu[1], "Hash params doivent différer CPU vs GPU"

        # Les hash de data doivent être identiques (même DataFrame)
        assert key_cpu[2] == key_gpu[2], "Hash data doivent être identiques"

        print(f"✅ Clé CPU: {key_cpu[0]}")
        print(f"✅ Clé GPU: {key_gpu[0]}")

    def test_cache_auto_detects_backend_small_data(self):
        """
        Test : Avec petites données (< 5000), le backend doit être 'cpu'.
        """
        bank = IndicatorBank()
        df = create_sample_data(n_bars=1000)  # < 5000 → CPU

        key = bank._generate_key("rsi", {"period": 14}, df)

        # La clé doit contenir "_backend": "cpu" dans le hash
        # On peut vérifier en re-générant avec backend explicite
        key_cpu_explicit = bank._generate_key("rsi", {"period": 14, "_backend": "cpu"}, df)

        assert key[0] == key_cpu_explicit[0], "Petites données → auto-détection CPU"
        print(f"✅ Auto-détection CPU (1000 bars): {key[0]}")

    def test_cache_auto_detects_backend_large_data(self):
        """
        Test : Avec grandes données (>= 5000), le backend doit être 'gpu' si disponible.
        """
        bank = IndicatorBank()
        df = create_sample_data(n_bars=10000)  # >= 5000 → GPU si disponible

        key = bank._generate_key("rsi", {"period": 14}, df)

        # Déterminer quel backend devrait être utilisé
        try:
            from performance.gpu import gpu_available
            expected_backend = "gpu" if gpu_available() else "cpu"
        except Exception:
            expected_backend = "cpu"

        key_expected = bank._generate_key(
            "rsi",
            {"period": 14, "_backend": expected_backend},
            df
        )

        assert key[0] == key_expected[0], f"Grandes données → auto-détection {expected_backend}"
        print(f"✅ Auto-détection {expected_backend} (10000 bars): {key[0]}")

    def test_cache_no_collision_same_params_diff_backend(self):
        """
        Test : Scénario de bug réel - éviter collision cache CPU/GPU.

        Scénario:
        1. Run 1: GPU activé (float32) → cache RSI
        2. Run 2: GPU désactivé (float64) → ne doit PAS réutiliser cache GPU
        """
        bank = IndicatorBank()
        df = create_sample_data(n_bars=10000)

        params_rsi = {"period": 14}

        # Simuler résultat GPU (float32) - UTILISER le paramètre backend=
        result_gpu = np.array([30.5, 45.2, 60.1], dtype=np.float32)
        bank.put("rsi", params_rsi, df, result_gpu, backend="gpu")

        # Tentative de récupération avec backend CPU
        cached_cpu = bank.get("rsi", params_rsi, df, backend="cpu")

        # Ne doit PAS récupérer le résultat GPU
        assert cached_cpu is None, "Cache GPU ne doit pas être utilisé pour CPU"

        print("✅ Pas de collision cache CPU/GPU")

    def test_cache_preserves_dtype(self):
        """
        Test : Le cache doit préserver le dtype (float32 vs float64).
        """
        bank = IndicatorBank()
        df = create_sample_data(n_bars=1000)

        # Mettre en cache un résultat float32 (GPU) - UTILISER backend=
        result_gpu = np.array([30.5, 45.2, 60.1], dtype=np.float32)
        bank.put("rsi", {"period": 14}, df, result_gpu, backend="gpu")

        # Récupérer avec même backend
        cached = bank.get("rsi", {"period": 14}, df, backend="gpu")

        assert cached is not None, "Résultat doit être en cache"
        assert cached.dtype == np.float32, "dtype doit être préservé (float32)"

        # Mettre en cache un résultat float64 (CPU)
        result_cpu = np.array([30.5, 45.2, 60.1], dtype=np.float64)
        bank.put("rsi", {"period": 14}, df, result_cpu, backend="cpu")

        # Récupérer avec backend CPU
        cached_cpu = bank.get("rsi", {"period": 14}, df, backend="cpu")

        assert cached_cpu is not None, "Résultat CPU doit être en cache"
        assert cached_cpu.dtype == np.float64, "dtype doit être préservé (float64)"

        print("✅ dtypes préservés (float32 GPU, float64 CPU)")


if __name__ == '__main__':
    # Exécuter les tests
    test = TestIndicatorBankCache()

    print("=" * 70)
    print("TESTS DU CACHE INDICATORBANK (CPU vs GPU)")
    print("=" * 70)

    try:
        test.test_cache_key_differs_by_backend()
        print("\n[1/5] ✅ test_cache_key_differs_by_backend")
    except AssertionError as e:
        print(f"\n[1/5] ❌ test_cache_key_differs_by_backend: {e}")

    try:
        test.test_cache_auto_detects_backend_small_data()
        print("\n[2/5] ✅ test_cache_auto_detects_backend_small_data")
    except AssertionError as e:
        print(f"\n[2/5] ❌ test_cache_auto_detects_backend_small_data: {e}")

    try:
        test.test_cache_auto_detects_backend_large_data()
        print("\n[3/5] ✅ test_cache_auto_detects_backend_large_data")
    except AssertionError as e:
        print(f"\n[3/5] ❌ test_cache_auto_detects_backend_large_data: {e}")

    try:
        test.test_cache_no_collision_same_params_diff_backend()
        print("\n[4/5] ✅ test_cache_no_collision_same_params_diff_backend")
    except AssertionError as e:
        print(f"\n[4/5] ❌ test_cache_no_collision_same_params_diff_backend: {e}")

    try:
        test.test_cache_preserves_dtype()
        print("\n[5/5] ✅ test_cache_preserves_dtype")
    except AssertionError as e:
        print(f"\n[5/5] ❌ test_cache_preserves_dtype: {e}")

    print("\n" + "=" * 70)
    print("TOUS LES TESTS TERMINÉS")
    print("=" * 70)
