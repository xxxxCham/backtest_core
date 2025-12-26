"""
Backtest Core - Parallel Processing Module
==========================================

Parallélisation des backtests avec joblib et multiprocessing.
Supporte le traitement de grilles de paramètres en parallèle.

Usage:
    >>> from performance.parallel import ParallelRunner, parallel_sweep
    >>>
    >>> # Option 1: Fonction simple
    >>> results = parallel_sweep(run_backtest, param_grid, n_jobs=-1)
    >>>
    >>> # Option 2: Classe complète
    >>> runner = ParallelRunner(max_workers=8)
    >>> results = runner.run_sweep(strategy, data, param_grid)
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

# Joblib pour parallélisation simple (optionnel)
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# psutil pour monitoring ressources (optionnel)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration pour l'exécution parallèle."""
    max_workers: int = -1  # -1 = auto (nb CPU)
    use_processes: bool = True  # True=multiprocessing, False=threading
    chunk_size: int = 10  # Taille des batches
    timeout: Optional[float] = None  # Timeout par tâche (secondes)
    memory_limit_gb: Optional[float] = None  # Limite mémoire


@dataclass
class SweepResult:
    """Résultat d'un sweep parallèle."""
    results: List[Dict[str, Any]]
    total_time: float
    n_completed: int
    n_failed: int
    avg_time_per_task: float
    memory_peak_gb: Optional[float] = None


def _get_cpu_count() -> int:
    """Retourne le nombre de CPUs disponibles."""
    if HAS_PSUTIL:
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    return os.cpu_count() or 4


def _get_available_memory_gb() -> float:
    """Retourne la mémoire disponible en GB."""
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass
    return 8.0  # Valeur par défaut


def generate_param_grid(param_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Génère toutes les combinaisons de paramètres.
    
    Args:
        param_ranges: Dict avec {param_name: [values]} ou {param_name: value}
        
    Returns:
        Liste de dicts, chaque dict étant une combinaison de paramètres
        
    Example:
        >>> grid = generate_param_grid({
        ...     "period": [10, 20, 30],
        ...     "threshold": [0.5, 1.0],
        ...     "leverage": 1  # Valeur fixe
        ... })
        >>> len(grid)  # 3 * 2 = 6 combinaisons
        6
    """
    import itertools
    
    # Normaliser: convertir valeurs scalaires en listes
    normalized = {}
    for key, value in param_ranges.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            normalized[key] = list(value)
        else:
            normalized[key] = [value]
    
    # Générer le produit cartésien
    keys = list(normalized.keys())
    values = list(normalized.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def parallel_sweep(
    func: Callable,
    param_grid: List[Dict[str, Any]],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    **fixed_kwargs
) -> List[Any]:
    """
    Exécute une fonction sur une grille de paramètres en parallèle.
    
    Utilise joblib si disponible, sinon ProcessPoolExecutor.
    
    Args:
        func: Fonction à appeler, signature: func(params, **fixed_kwargs)
        param_grid: Liste de dicts de paramètres
        n_jobs: Nombre de workers (-1 = tous les CPUs)
        backend: Backend joblib ('loky', 'multiprocessing', 'threading')
        verbose: Niveau de verbosité (0-10)
        **fixed_kwargs: Arguments fixes passés à chaque appel
        
    Returns:
        Liste des résultats dans le même ordre que param_grid
        
    Example:
        >>> def run_backtest(params, data=None):
        ...     return {"params": params, "sharpe": 1.5}
        >>> 
        >>> grid = [{"period": 10}, {"period": 20}]
        >>> results = parallel_sweep(run_backtest, grid, data=df)
    """
    if n_jobs == -1:
        n_jobs = _get_cpu_count()
    
    if HAS_JOBLIB:
        # Utiliser joblib (plus robuste et optimisé)
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(params, **fixed_kwargs) for params in param_grid
        )
        return results
    else:
        # Fallback sur concurrent.futures
        logger.info("joblib non disponible, utilisation de ProcessPoolExecutor")
        results = [None] * len(param_grid)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(func, params, **fixed_kwargs): i
                for i, params in enumerate(param_grid)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Erreur tâche {idx}: {e}")
                    results[idx] = {"error": str(e)}
        
        return results


class ParallelRunner:
    """
    Gestionnaire de backtests parallèles avec monitoring et optimisation.
    
    Supporte:
    - Exécution multi-processus ou multi-thread
    - Chunking automatique pour gestion mémoire
    - Monitoring CPU/RAM en temps réel
    - Arrêt anticipé sur critère
    
    Example:
        >>> runner = ParallelRunner(max_workers=8)
        >>> 
        >>> param_grid = generate_param_grid({
        ...     "bb_period": range(15, 35, 5),
        ...     "atr_mult": [1.5, 2.0, 2.5]
        ... })
        >>> 
        >>> results = runner.run_sweep(
        ...     strategy=my_strategy,
        ...     data=df,
        ...     param_grid=param_grid
        ... )
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: int = 50,
        memory_limit_gb: Optional[float] = None,
    ):
        """
        Initialise le runner parallèle.
        
        Args:
            max_workers: Nombre de workers (None = auto)
            use_processes: True=multiprocessing, False=threading
            chunk_size: Taille des batches pour gestion mémoire
            memory_limit_gb: Limite mémoire (None = pas de limite)
        """
        self.max_workers = max_workers or self._calculate_optimal_workers()
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        
        # État
        self._stop_requested = False
        self._current_progress = 0
        self._total_tasks = 0
        
        # Callbacks
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        
        logger.info(
            f"ParallelRunner initialisé: {self.max_workers} workers, "
            f"mode={'processes' if use_processes else 'threads'}, "
            f"chunk_size={chunk_size}"
        )
    
    def _calculate_optimal_workers(self) -> int:
        """Calcule le nombre optimal de workers."""
        cpu_count = _get_cpu_count()
        available_ram = _get_available_memory_gb()
        
        # Estimation: ~500MB par worker de backtest
        ram_limited_workers = int(available_ram / 0.5)
        
        optimal = min(cpu_count, ram_limited_workers)
        return max(1, optimal)
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Définit un callback de progression: callback(completed, total)."""
        self._progress_callback = callback
    
    def request_stop(self):
        """Demande l'arrêt du sweep en cours."""
        self._stop_requested = True
        logger.info("Arrêt demandé pour le sweep en cours")
    
    def _chunk_grid(self, param_grid: List[Dict]) -> Iterator[List[Dict]]:
        """Divise la grille en chunks pour gestion mémoire."""
        for i in range(0, len(param_grid), self.chunk_size):
            yield param_grid[i:i + self.chunk_size]
    
    def run_sweep(
        self,
        run_func: Callable,
        param_grid: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **fixed_kwargs
    ) -> SweepResult:
        """
        Exécute un sweep parallèle complet.
        
        Args:
            run_func: Fonction de backtest, signature: run_func(params, **kwargs)
            param_grid: Liste des combinaisons de paramètres
            progress_callback: Callback optionnel (completed, total)
            **fixed_kwargs: Arguments fixes (data, etc.)
            
        Returns:
            SweepResult avec tous les résultats et métriques
        """
        self._stop_requested = False
        self._total_tasks = len(param_grid)
        self._current_progress = 0
        
        if progress_callback:
            self._progress_callback = progress_callback
        
        start_time = time.time()
        all_results = []
        n_failed = 0
        memory_peak = 0.0
        
        # Choisir l'executor
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        logger.info(
            f"Démarrage sweep: {self._total_tasks} tâches, "
            f"{self.max_workers} workers"
        )
        
        # Traitement par chunks
        for chunk in self._chunk_grid(param_grid):
            if self._stop_requested:
                logger.info("Sweep arrêté par demande utilisateur")
                break
            
            # Vérifier mémoire si limite définie
            if self.memory_limit_gb and HAS_PSUTIL:
                current_mem = psutil.virtual_memory().used / (1024 ** 3)
                if current_mem > self.memory_limit_gb:
                    logger.warning(f"Limite mémoire atteinte: {current_mem:.1f} GB")
                    break
            
            # Exécuter le chunk
            with ExecutorClass(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(run_func, params, **fixed_kwargs): params
                    for params in chunk
                }
                
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        result = future.result(timeout=300)  # 5min timeout
                        all_results.append({
                            "params": params,
                            "result": result,
                            "success": True
                        })
                    except Exception as e:
                        logger.error(f"Erreur: {params} -> {e}")
                        all_results.append({
                            "params": params,
                            "error": str(e),
                            "success": False
                        })
                        n_failed += 1
                    
                    self._current_progress += 1
                    if self._progress_callback:
                        self._progress_callback(self._current_progress, self._total_tasks)
            
            # Tracking mémoire
            if HAS_PSUTIL:
                current_mem = psutil.virtual_memory().used / (1024 ** 3)
                memory_peak = max(memory_peak, current_mem)
        
        elapsed = time.time() - start_time
        n_completed = len([r for r in all_results if r.get("success")])
        
        return SweepResult(
            results=all_results,
            total_time=elapsed,
            n_completed=n_completed,
            n_failed=n_failed,
            avg_time_per_task=elapsed / max(1, len(all_results)),
            memory_peak_gb=memory_peak if memory_peak > 0 else None
        )


def run_backtest_worker(
    params: Dict[str, Any],
    strategy_class: type,
    data: pd.DataFrame,
    indicators: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Worker function pour exécuter un backtest (picklable).
    
    Args:
        params: Paramètres de la stratégie
        strategy_class: Classe de stratégie à instancier
        data: DataFrame OHLCV
        indicators: Indicateurs précalculés
        
    Returns:
        Dict avec params et métriques de performance
    """
    try:
        strategy = strategy_class()
        result = strategy.run(data, indicators, params)
        
        # Calculer métriques simples
        signals = result.signals
        n_trades = int(np.sum(signals != 0))
        
        return {
            "params": params,
            "n_trades": n_trades,
            "success": True
        }
    except Exception as e:
        return {
            "params": params,
            "error": str(e),
            "success": False
        }


# ======================== Utilitaires de benchmark ========================

def benchmark_parallel_configs(
    func: Callable,
    sample_params: List[Dict],
    configs: Optional[List[ParallelConfig]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark différentes configurations parallèles.
    
    Args:
        func: Fonction à tester
        sample_params: Échantillon de paramètres
        configs: Liste de configurations à tester (None = auto)
        **kwargs: Arguments fixes pour func
        
    Returns:
        Dict avec meilleure config et résultats de benchmark
    """
    if configs is None:
        cpu_count = _get_cpu_count()
        configs = [
            ParallelConfig(max_workers=cpu_count // 2, use_processes=True),
            ParallelConfig(max_workers=cpu_count, use_processes=True),
            ParallelConfig(max_workers=cpu_count * 2, use_processes=True),
            ParallelConfig(max_workers=cpu_count, use_processes=False),  # threading
        ]
    
    results = []
    
    for config in configs:
        runner = ParallelRunner(
            max_workers=config.max_workers,
            use_processes=config.use_processes,
            chunk_size=config.chunk_size
        )
        
        start = time.time()
        runner.run_sweep(func, sample_params[:min(20, len(sample_params))], **kwargs)
        elapsed = time.time() - start
        
        results.append({
            "config": config,
            "elapsed": elapsed,
            "throughput": len(sample_params) / elapsed if elapsed > 0 else 0
        })
        
        logger.info(
            f"Config: {config.max_workers} workers, "
            f"{'process' if config.use_processes else 'thread'} -> "
            f"{elapsed:.2f}s ({results[-1]['throughput']:.1f} tâches/s)"
        )
    
    # Trouver la meilleure config
    best = max(results, key=lambda x: x["throughput"])
    
    return {
        "best_config": best["config"],
        "best_throughput": best["throughput"],
        "all_results": results
    }
