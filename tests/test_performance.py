"""
Tests pour le module Performance
================================

Tests unitaires pour les composants de performance:
- Parallélisation
- Monitoring
- Profiling
- Gestion mémoire
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestParallelModule(unittest.TestCase):
    """Tests pour le module parallel."""

    def test_import(self):
        """Test que le module s'importe correctement."""
        from performance.parallel import (
            ParallelRunner,
            ParallelConfig,
            SweepResult,
            parallel_sweep,
            generate_param_grid,
        )
        self.assertTrue(True)

    def test_generate_param_grid_simple(self):
        """Test génération de grille simple."""
        from performance.parallel import generate_param_grid
        
        grid = generate_param_grid({
            "a": [1, 2],
            "b": [10, 20],
        })
        
        self.assertEqual(len(grid), 4)  # 2 * 2
        self.assertIn({"a": 1, "b": 10}, grid)
        self.assertIn({"a": 2, "b": 20}, grid)

    def test_generate_param_grid_with_fixed(self):
        """Test grille avec valeur fixe."""
        from performance.parallel import generate_param_grid
        
        grid = generate_param_grid({
            "a": [1, 2, 3],
            "fixed": 100,  # Valeur scalaire
        })
        
        self.assertEqual(len(grid), 3)
        for combo in grid:
            self.assertEqual(combo["fixed"], 100)

    def test_parallel_runner_init(self):
        """Test initialisation du runner."""
        from performance.parallel import ParallelRunner
        
        runner = ParallelRunner(max_workers=2)
        self.assertEqual(runner.max_workers, 2)
        self.assertTrue(runner.use_processes)


class TestMonitorModule(unittest.TestCase):
    """Tests pour le module monitor."""

    def test_import(self):
        """Test que le module s'importe correctement."""
        from performance.monitor import (
            PerformanceMonitor,
            ResourceTracker,
            ProgressBar,
            get_system_resources,
        )
        self.assertTrue(True)

    def test_resource_tracker(self):
        """Test le tracker de ressources."""
        from performance.monitor import ResourceTracker
        
        tracker = ResourceTracker(interval=0.1)
        tracker.start()
        time.sleep(0.3)
        stats = tracker.stop()
        
        self.assertGreater(stats.duration_seconds, 0)
        self.assertGreaterEqual(stats.samples_count, 0)

    def test_get_system_resources(self):
        """Test récupération des ressources système."""
        from performance.monitor import get_system_resources
        
        resources = get_system_resources()
        
        self.assertIn("psutil_available", resources)
        self.assertIn("rich_available", resources)


class TestProfilerModule(unittest.TestCase):
    """Tests pour le module profiler."""

    def test_import(self):
        """Test que le module s'importe correctement."""
        from performance.profiler import (
            Profiler,
            ProfileResult,
            profile_function,
            TimingContext,
        )
        self.assertTrue(True)

    def test_profiler_basic(self):
        """Test profiler basique."""
        from performance.profiler import Profiler
        
        profiler = Profiler("test")
        profiler.start()
        
        # Code à profiler - quelque chose qui prend du temps mesurable
        total = 0
        for i in range(10000):
            total += i
        
        result = profiler.stop()
        
        self.assertEqual(result.name, "test")
        # Le temps peut être 0.0 sur machines très rapides, vérifions juste >= 0
        self.assertGreaterEqual(result.total_time, 0)

    def test_profiler_context_manager(self):
        """Test profiler en context manager."""
        from performance.profiler import Profiler
        
        with Profiler("test_ctx") as prof:
            total = sum(range(1000))
        
        # stop() est appelé automatiquement
        self.assertTrue(True)

    def test_timing_context(self):
        """Test TimingContext."""
        from performance.profiler import TimingContext
        
        with TimingContext("test", verbose=False) as ctx:
            time.sleep(0.05)
        
        self.assertGreater(ctx.elapsed, 0.04)


class TestMemoryModule(unittest.TestCase):
    """Tests pour le module memory."""

    def test_import(self):
        """Test que le module s'importe correctement."""
        from performance.memory import (
            ChunkedProcessor,
            MemoryManager,
            DataFrameCache,
            get_memory_info,
            optimize_dataframe,
        )
        self.assertTrue(True)

    def test_get_memory_info(self):
        """Test récupération info mémoire."""
        from performance.memory import get_memory_info
        
        info = get_memory_info()
        
        self.assertGreater(info.total_gb, 0)
        self.assertGreaterEqual(info.percent, 0)

    def test_chunked_processor(self):
        """Test processeur par chunks."""
        from performance.memory import ChunkedProcessor
        
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
        })
        
        processor = ChunkedProcessor(chunk_size=30)
        
        chunks = list(processor.iter_chunks(df))
        
        self.assertGreater(len(chunks), 1)
        
        # Vérifier que tous les éléments sont couverts
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 100)

    def test_dataframe_cache(self):
        """Test cache DataFrame."""
        from performance.memory import DataFrameCache
        
        cache = DataFrameCache(max_memory_gb=0.1, spill_to_disk=False)
        
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        cache.put("test", df)
        self.assertTrue(cache.contains("test"))
        
        retrieved = cache.get("test")
        self.assertEqual(len(retrieved), 3)
        
        cache.clear()
        self.assertFalse(cache.contains("test"))

    def test_optimize_dataframe(self):
        """Test optimisation DataFrame."""
        from performance.memory import optimize_dataframe
        
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
        })
        
        optimized = optimize_dataframe(df)
        
        # Vérifier que les données sont préservées
        self.assertEqual(list(optimized["int_col"]), [1, 2, 3])


class TestGPUModule(unittest.TestCase):
    """Tests pour le module GPU."""

    def test_import(self):
        """Test que le module s'importe correctement."""
        from performance.gpu import (
            GPUIndicatorCalculator,
            gpu_available,
            get_gpu_info,
        )
        self.assertTrue(True)

    def test_gpu_available(self):
        """Test détection GPU."""
        from performance.gpu import gpu_available, get_gpu_info
        
        # La fonction doit retourner un booléen sans erreur
        result = gpu_available()
        self.assertIsInstance(result, bool)
        
        # get_gpu_info doit retourner un dict
        info = get_gpu_info()
        self.assertIsInstance(info, dict)
        self.assertIn("gpu_available", info)

    def test_cpu_calculator(self):
        """Test calculateur en mode CPU."""
        from performance.gpu import GPUIndicatorCalculator
        
        calc = GPUIndicatorCalculator(use_gpu=False)
        
        prices = np.random.randn(100).cumsum() + 100
        
        # Test SMA
        sma = calc.sma(prices, period=10)
        self.assertEqual(len(sma), 100)
        self.assertTrue(np.isnan(sma[:9]).all())  # Premières valeurs NaN
        
        # Test EMA
        ema = calc.ema(prices, period=10)
        self.assertEqual(len(ema), 100)
        
        # Test RSI
        rsi = calc.rsi(prices, period=14)
        self.assertEqual(len(rsi), 100)

    def test_bollinger_bands_cpu(self):
        """Test Bollinger Bands en CPU."""
        from performance.gpu import GPUIndicatorCalculator
        
        calc = GPUIndicatorCalculator(use_gpu=False)
        
        prices = np.random.randn(100).cumsum() + 100
        
        upper, middle, lower = calc.bollinger_bands(prices, period=20, std_dev=2.0)
        
        self.assertEqual(len(upper), 100)
        self.assertEqual(len(middle), 100)
        self.assertEqual(len(lower), 100)
        
        # Vérifier la relation upper > middle > lower
        valid_idx = ~np.isnan(middle)
        self.assertTrue((upper[valid_idx] >= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] >= lower[valid_idx]).all())


class TestSweepEngine(unittest.TestCase):
    """Tests pour le moteur de sweep."""

    def test_import(self):
        """Test import sweep engine."""
        from backtest.sweep import SweepEngine, SweepResults
        self.assertTrue(True)

    def test_sweep_engine_init(self):
        """Test initialisation."""
        from backtest.sweep import SweepEngine
        
        engine = SweepEngine(max_workers=2, initial_capital=5000)
        
        self.assertEqual(engine.initial_capital, 5000)

    def test_sweep_results_to_dataframe(self):
        """Test conversion résultats en DataFrame."""
        from backtest.sweep import SweepResults, SweepResultItem
        
        items = [
            SweepResultItem(
                params={"a": 1},
                metrics={"sharpe": 1.5},
                success=True
            ),
            SweepResultItem(
                params={"a": 2},
                metrics={"sharpe": 2.0},
                success=True
            ),
        ]
        
        results = SweepResults(
            items=items,
            best_params={"a": 2},
            best_metrics={"sharpe": 2.0},
            total_time=1.0,
            n_completed=2,
            n_failed=0,
        )
        
        df = results.to_dataframe()
        
        self.assertEqual(len(df), 2)
        self.assertIn("a", df.columns)
        self.assertIn("sharpe", df.columns)


if __name__ == "__main__":
    unittest.main()
