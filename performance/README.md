# Performance Module - Guide Utilisateur

> **Module d'optimisation des performances pour backtest_core**  
> Version : 1.8.0 | Date : 13/12/2025

---

## 📊 Vue d'Ensemble

Le module `performance/` fournit des outils d'optimisation pour le moteur de backtest :
- ⚡ **Parallélisation CPU** : Distribuer les calculs sur plusieurs cœurs
- 🚀 **Accélération GPU** : Utiliser CuPy/Numba pour calculs massifs
- 📈 **Monitoring temps réel** : Surveiller CPU/RAM/GPU pendant l'exécution
- 🔍 **Profiling** : Identifier les goulots d'étranglement
- 💾 **Gestion mémoire** : Optimiser l'utilisation RAM avec chunking

---

## 📁 Structure

```
performance/
├── parallel.py         → Parallélisation CPU (joblib/multiprocessing)
├── monitor.py          → Monitoring temps réel (psutil + rich)
├── profiler.py         → Profiling cProfile + line_profiler
├── memory.py           → Gestion mémoire + chunking
├── benchmark.py        → Suite de benchmarks v1.8.0
├── device_backend.py   → Abstraction NumPy/CuPy transparente
├── gpu.py              → Utilitaires GPU
└── __init__.py         → Exports publics
```

---

## 🚀 Guide d'Utilisation

### 1. Parallélisation (`parallel.py`)

**Objectif :** Accélérer les sweeps en distribuant les calculs sur plusieurs CPU.

```python
from performance import ParallelRunner, parallel_sweep

# Méthode 1: Runner avec configuration
runner = ParallelRunner(n_jobs=8)
results = runner.run_sweep(strategy, param_grid, data)

# Méthode 2: Fonction directe
results = parallel_sweep(
    strategy_class=EMACrossStrategy,
    param_grid={"fast": [5, 10, 15], "slow": [20, 30, 50]},
    data=ohlcv_df,
    n_jobs=8,
)

# Benchmark différentes configs
benchmark_parallel_configs(strategy, param_grid, data)
```

**Classes principales :**
- `ParallelRunner` : Exécuteur parallèle configurable
- `ParallelConfig` : Configuration (n_jobs, backend, timeout)
- `SweepResult` : Résultat d'un sweep parallèle

**Performances typiques :**
| Workers | Speedup | CPU Usage |
|---------|---------|-----------|
| 1 | 1.0x | 12.5% |
| 4 | 3.2x | 50% |
| 8 | 5.8x | 100% |

---

### 2. Monitoring (`monitor.py`)

**Objectif :** Surveiller les ressources système en temps réel pendant les backtests.

```python
from performance import PerformanceMonitor, ProgressBar

# Monitor avec rich console
with PerformanceMonitor(show_bar=True) as monitor:
    for params in param_grid:
        result = engine.run(params)
        monitor.update(1)  # Avancer la barre

# Progress bar standalone
with ProgressBar(total=len(param_grid)) as pbar:
    for i, params in enumerate(param_grid):
        result = engine.run(params)
        pbar.update(1, description=f"Params {i+1}")

# Stats système
from performance import get_system_resources, print_system_info

stats = get_system_resources()  # CPU, RAM, GPU stats
print_system_info()             # Affichage formaté
```

**Classes principales :**
- `PerformanceMonitor` : Monitor avec barre de progression
- `ResourceTracker` : Tracking CPU/RAM/GPU en continu
- `ProgressBar` : Barre de progression rich

---

### 3. Profiling (`profiler.py`)

**Objectif :** Identifier les parties lentes du code.

```python
from performance import Profiler, profile_function, TimingContext

# Méthode 1: Context manager
with Profiler() as profiler:
    engine.run_sweep(param_grid)

profiler.print_stats()
profiler.save_stats("profile.prof")

# Méthode 2: Décorateur
@profile_function
def my_backtest_function(params):
    return engine.run(params)

# Méthode 3: Timing simple
with TimingContext("Calcul indicateurs"):
    indicators = compute_indicators(data)
# Output: "Calcul indicateurs: 1.234s"

# Benchmark une fonction
from performance import benchmark_function

stats = benchmark_function(
    lambda: engine.run(params),
    n_runs=100,
    warmup=10,
)
print(f"Moyenne: {stats.mean_time:.4f}s")
```

**Classes principales :**
- `Profiler` : Wrapper cProfile + line_profiler
- `ProfileResult` : Résultat d'un profiling
- `TimingContext` : Chronométrage simple

---

### 4. Gestion Mémoire (`memory.py`)

**Objectif :** Optimiser l'utilisation RAM pour datasets volumineux.

```python
from performance import (
    ChunkedProcessor,
    MemoryManager,
    DataFrameCache,
    optimize_dataframe,
    memory_efficient_mode,
)

# Méthode 1: Chunking automatique
processor = ChunkedProcessor(chunk_size_mb=100)
results = processor.process_dataframe(large_df, compute_function)

# Méthode 2: Cache LRU
cache = DataFrameCache(max_size_gb=2.0)
cached_df = cache.get_or_compute("key", lambda: load_heavy_data())

# Méthode 3: Optimisation DataFrame
df_optimized = optimize_dataframe(df)  # Réduire mémoire 30-70%

# Context manager mode économie
with memory_efficient_mode():
    # Limite RAM utilisée automatiquement
    results = heavy_computation()

# Stats mémoire
from performance import get_memory_info, get_available_ram_gb

mem_stats = get_memory_info()
print(f"RAM libre: {get_available_ram_gb():.2f} GB")
```

**Classes principales :**
- `ChunkedProcessor` : Découpage automatique datasets
- `MemoryManager` : Gestion mémoire globale
- `DataFrameCache` : Cache LRU avec limite GB

---

### 5. GPU (`gpu.py` + `device_backend.py`)

**Objectif :** Accélérer calculs avec GPU (CuPy) tout en gardant compatibilité CPU.

```python
from performance import (
    GPUIndicatorCalculator,
    gpu_available,
    get_gpu_info,
    to_gpu,
    to_cpu,
)

# Check disponibilité GPU
if gpu_available():
    print(get_gpu_info())
else:
    print("GPU non disponible, fallback CPU")

# Calcul GPU transparent
calc = GPUIndicatorCalculator()
result = calc.compute_ema(prices, period=20)  # Auto GPU si dispo

# Transfer manuel CPU ↔ GPU
gpu_array = to_gpu(cpu_array)  # NumPy → CuPy
cpu_array = to_cpu(gpu_array)  # CuPy → NumPy

# Backend agnostic (v1.8.0)
from performance.device_backend import ArrayBackend

backend = ArrayBackend.auto()  # Détecte NumPy ou CuPy
arr = backend.array([1, 2, 3])
result = backend.mean(arr)
```

---

### 6. Benchmark Suite (`benchmark.py`)

**Objectif :** Comparer performances de différentes implémentations.

```python
from performance.benchmark import (
    run_all_benchmarks,
    benchmark_indicator_calculation,
    benchmark_simulator_performance,
    benchmark_gpu_vs_cpu,
)

# Benchmark complet
run_all_benchmarks(verbose=True)

# Benchmark spécifique
comp = benchmark_indicator_calculation(data_size=10000)
print(comp.summary())

# GPU vs CPU
comp = benchmark_gpu_vs_cpu(data_size=100000)
print(f"Speedup GPU: {comp.speedup:.2f}x")
```

---

## 🎯 Cas d'Usage Typiques

### Cas 1 : Sweep Rapide avec Monitoring

```python
from performance import parallel_sweep, PerformanceMonitor

with PerformanceMonitor(show_bar=True) as monitor:
    results = parallel_sweep(
        strategy_class=EMACrossStrategy,
        param_grid=large_grid,
        data=df,
        n_jobs=8,
    )

print(f"Meilleur Sharpe: {max(r.sharpe for r in results)}")
```

### Cas 2 : Backtest GPU avec Gestion Mémoire

```python
from performance import GPUIndicatorCalculator, memory_efficient_mode

with memory_efficient_mode():
    calc = GPUIndicatorCalculator()
    indicators = calc.compute_all(data)
    results = engine.run(indicators)
```

### Cas 3 : Profiling d'une Stratégie Lente

```python
from performance import Profiler

with Profiler() as profiler:
    engine.run_sweep(param_grid)

profiler.print_stats(top=10)
# Output: Top 10 des fonctions les plus lentes
```

---

## 📊 Dépendances Optionnelles

| Package | Usage | Installation |
|---------|-------|--------------|
| `joblib` | Parallélisation | `pip install joblib` |
| `psutil` | Monitoring système | `pip install psutil` |
| `rich` | Console formatée | `pip install rich` |
| `cupy` | Accélération GPU | `pip install cupy-cuda12x` |
| `line_profiler` | Profiling ligne par ligne | `pip install line_profiler` |
| `numba` | JIT compilation | `pip install numba` |

**Note :** Toutes les fonctionnalités ont un **fallback gracieux** si la dépendance est absente.

---

## 🔧 Configuration

Variables d'environnement disponibles :

```bash
# GPU
BACKTEST_USE_GPU=True                    # Activer GPU (défaut: False)
CUPY_CACHE_DIR=/path/to/cache            # Cache CuPy

# Parallélisation
BACKTEST_N_JOBS=8                        # Workers par défaut
BACKTEST_PARALLEL_BACKEND=multiprocessing # joblib ou multiprocessing

# Mémoire
BACKTEST_MEMORY_LIMIT_GB=16.0            # Limite RAM
BACKTEST_CHUNK_SIZE_MB=100               # Taille chunks
```

---

## 📈 Performances Attendues

| Opération | Sans Optimisation | Avec Optimisation | Speedup |
|-----------|-------------------|-------------------|---------|
| Sweep 1000 params | 120s | 21s (8 workers) | **5.7x** |
| Calcul indicateurs | 5s | 0.3s (GPU) | **16.6x** |
| Backtest 10M rows | OOM | 45s (chunking) | ✅ Fonctionne |

---

## 🐛 Troubleshooting

### Erreur : "No module named 'cupy'"
```bash
pip install cupy-cuda12x  # CUDA 12.x
# Ou désactiver GPU
BACKTEST_USE_GPU=False python script.py
```

### Erreur : "MemoryError"
```python
from performance import memory_efficient_mode

# Activer mode économie mémoire
with memory_efficient_mode():
    results = heavy_computation()
```

### Parallélisation lente
```python
# Vérifier overhead communication
from performance import benchmark_parallel_configs

benchmark_parallel_configs(strategy, param_grid, data)
# Choisir le meilleur nombre de workers
```

---

## 📚 Références

- [joblib documentation](https://joblib.readthedocs.io/)
- [CuPy user guide](https://docs.cupy.dev/en/stable/user_guide/)
- [Numba JIT guide](https://numba.pydata.org/numba-doc/latest/user/jit.html)
- [psutil documentation](https://psutil.readthedocs.io/)

---

*Dernière mise à jour : 13/12/2025 | Version : 1.8.0*
