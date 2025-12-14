"""
Backtest Core - Performance Module
==================================

Module d'optimisation des performances pour le moteur de backtest.

Inclut:
- Parallélisation CPU (joblib/multiprocessing)
- Accélération GPU (cupy/numba - optionnel)
- Monitoring temps réel (rich)
- Profiling intégré (cProfile)
- Gestion mémoire (chunking)

Usage:
    >>> from performance import ParallelRunner, parallel_sweep
    >>> from performance import PerformanceMonitor, ProgressBar
    >>> from performance import Profiler, profile_function
    >>> from performance import ChunkedProcessor, MemoryManager
    >>> from performance import GPUIndicatorCalculator, gpu_available
"""

from performance.parallel import (
    ParallelRunner,
    ParallelConfig,
    SweepResult,
    parallel_sweep,
    generate_param_grid,
    benchmark_parallel_configs,
)

from performance.monitor import (
    PerformanceMonitor,
    ResourceTracker,
    ResourceStats,
    ProgressBar,
    print_system_info,
    get_system_resources,
)

from performance.profiler import (
    Profiler,
    ProfileResult,
    profile_function,
    profile_memory,
    TimingContext,
    run_with_profiling,
    benchmark_function,
)

from performance.memory import (
    ChunkedProcessor,
    MemoryManager,
    DataFrameCache,
    MemoryStats,
    get_memory_info,
    get_available_ram_gb,
    optimize_dataframe,
    estimate_memory_needed,
    memory_efficient_mode,
)

from performance.gpu import (
    GPUIndicatorCalculator,
    gpu_available,
    get_gpu_info,
    to_gpu,
    to_cpu,
    benchmark_gpu_cpu,
)

__all__ = [
    # Parallel
    "ParallelRunner",
    "ParallelConfig",
    "SweepResult",
    "parallel_sweep",
    "generate_param_grid",
    "benchmark_parallel_configs",
    # Monitor
    "PerformanceMonitor",
    "ResourceTracker",
    "ResourceStats",
    "ProgressBar",
    "print_system_info",
    "get_system_resources",
    # Profiler
    "Profiler",
    "ProfileResult",
    "profile_function",
    "profile_memory",
    "TimingContext",
    "run_with_profiling",
    "benchmark_function",
    # Memory
    "ChunkedProcessor",
    "MemoryManager",
    "DataFrameCache",
    "MemoryStats",
    "get_memory_info",
    "get_available_ram_gb",
    "optimize_dataframe",
    "estimate_memory_needed",
    "memory_efficient_mode",
    # GPU
    "GPUIndicatorCalculator",
    "gpu_available",
    "get_gpu_info",
    "to_gpu",
    "to_cpu",
    "benchmark_gpu_cpu",
]

