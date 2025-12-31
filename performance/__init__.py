"""
Module-ID: performance.__init__

Purpose: Package performance - exports profiler, parallel, monitor, GPU, benchmarks.

Role in pipeline: performance optimization & observability

Key components: Re-exports ParallelRunner, Profiler, PerformanceMonitor, GPUIndicatorCalculator

Inputs: None (module imports only)

Outputs: Public API via __all__

Dependencies: .profiler, .parallel, .monitor, .memory, .gpu, .device_backend, .benchmark

Conventions: __all__ définit API publique; imports conditionnels pour optional deps.

Read-if: Modification exports ou module structure.

Skip-if: Vous importez directement depuis performance.profiler.
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
