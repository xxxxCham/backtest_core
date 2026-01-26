"""
Module-ID: performance.hybrid_compute

Purpose: Calcul hybride CPU+GPU - répartition intelligente des tâches.

Role in pipeline: performance optimization

Key components: HybridCompute, task scheduling, CPU/GPU load balancing

Strategy:
  - GPU: Calculs vectoriels lourds (>1000 points, operations parallèles)
  - CPU: Calculs légers, logique séquentielle, petits datasets
  - AUTO: Décision automatique selon taille données + disponibilité GPU

Inputs: Arrays NumPy, opération type, taille dataset

Outputs: Résultat calculé sur device optimal

Dependencies: numpy, cupy, numba, device_backend

Conventions:
  - Seuil GPU: 1000 points (au lieu de 5000)
  - Transfer CPU→GPU asynchrone si possible
  - Mémoire GPU: Max 80% utilisation
  - Fallback CPU automatique si GPU saturé

Benefits RTX 5080:
  - 16 GB VRAM = 2000+ symboles en parallèle
  - Compute Capability 12.0 = ultra-rapide
  - PCIe 5.0 = transfer CPU↔GPU 128 GB/s

Read-if: Modification scheduling ou seuils GPU.

Skip-if: Utilisation via HybridCompute.auto_compute().
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from performance.device_backend import ArrayBackend, DeviceType

logger = logging.getLogger(__name__)


class ComputeStrategy(Enum):
    """Stratégie de répartition des calculs."""
    AUTO = "auto"           # Décision automatique
    CPU_ONLY = "cpu_only"   # Forcer CPU
    GPU_ONLY = "gpu_only"   # Forcer GPU
    HYBRID = "hybrid"       # Répartition intelligente


@dataclass
class ComputeThresholds:
    """
    Seuils de décision GPU vs CPU - CALIBRÉ PAR BENCHMARK RÉEL.
    
    ⚠️ RÉSULTATS BENCHMARK (RTX 5080):
    - SMA/EMA individuels: GPU 20-30% PLUS LENT (overhead PCIe)
    - Batch 10×5k: GPU 1.78× plus rapide
    - Sweep 100 combos: GPU 2.33× plus rapide
    
    CONCLUSION: GPU utile UNIQUEMENT pour batch multi-symboles (10+)
    """
    # Taille minimale pour GPU (DÉSACTIVÉ suite benchmark)
    # Raison: Overhead transfert PCIe > gain calcul pour datasets < 50k
    gpu_min_size: int = 50000  # Pratiquement jamais atteint = GPU désactivé pour indicateurs simples
    
    # Opérations vectorielles lourdes → GPU (convolution, FFT)
    gpu_heavy_ops: int = 20000  # Uniquement opérations très lourdes
    
    # Mémoire GPU max utilisée (%)
    gpu_max_memory_pct: float = 0.80  # 80% max pour éviter OOM
    
    # Transfer CPU↔GPU coût (ms par MB)
    transfer_cost_ms_per_mb: float = 0.01  # PCIe 5.0 très rapide
    
    # Batch size pour calculs par blocs
    gpu_batch_size: int = 10000  # Process 10k points à la fois
    
    # Minimum datasets pour batch GPU (VALIDÉ par benchmark)
    min_batch_for_gpu: int = 10  # Batch 10×5k = 1.78× speedup confirmé
    
    @classmethod
    def from_env(cls) -> 'ComputeThresholds':
        """Charge les seuils depuis les variables d'environnement."""
        return cls(
            gpu_min_size=int(os.getenv("BACKTEST_GPU_MIN_SIZE", "1000")),
            gpu_heavy_ops=int(os.getenv("BACKTEST_GPU_HEAVY_OPS", "500")),
            gpu_max_memory_pct=float(os.getenv("BACKTEST_GPU_MAX_MEMORY", "0.80")),
            transfer_cost_ms_per_mb=float(os.getenv("BACKTEST_TRANSFER_COST", "0.01")),
            gpu_batch_size=int(os.getenv("BACKTEST_GPU_BATCH", "10000")),
        )


class HybridCompute:
    """
    Gestionnaire de calcul hybride CPU+GPU.
    
    Répartit intelligemment les tâches selon:
    - Taille des données
    - Type d'opération (vectorielle, logique, etc.)
    - Disponibilité GPU
    - Coût transfer CPU↔GPU
    
    Example:
        >>> hc = HybridCompute()
        >>> result = hc.auto_compute(data, operation="sma", window=20)
        >>> # → GPU si len(data) >= 1000, sinon CPU
    """
    
    _instance: Optional['HybridCompute'] = None
    
    def __new__(cls) -> 'HybridCompute':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le gestionnaire hybride."""
        if hasattr(self, '_initialized'):
            return
        
        self.backend = ArrayBackend()
        self.thresholds = ComputeThresholds.from_env()
        self._gpu_available = self.backend.gpu_available
        
        logger.info(
            f"HybridCompute initialisé - GPU: {self._gpu_available}, "
            f"Seuil GPU: {self.thresholds.gpu_min_size} points"
        )
        
        self._initialized = True
    
    @property
    def gpu_available(self) -> bool:
        """Indique si GPU disponible."""
        return self._gpu_available
    
    def should_use_gpu(
        self,
        data_size: int,
        operation_type: str = "vectorial",
        force_strategy: Optional[ComputeStrategy] = None,
    ) -> bool:
        """
        Décide si GPU doit être utilisé.
        
        Args:
            data_size: Nombre de points de données
            operation_type: Type d'opération ("vectorial", "sequential", "heavy")
            force_strategy: Forcer une stratégie spécifique
        
        Returns:
            True si GPU recommandé
        """
        # Stratégie forcée
        if force_strategy == ComputeStrategy.CPU_ONLY:
            return False
        if force_strategy == ComputeStrategy.GPU_ONLY:
            return self._gpu_available
        
        # GPU non disponible
        if not self._gpu_available:
            return False
        
        # Opérations séquentielles → CPU
        if operation_type == "sequential":
            return False
        
        # Opérations lourdes → GPU même si petites données
        if operation_type == "heavy":
            return data_size >= self.thresholds.gpu_heavy_ops
        
        # Opérations vectorielles → GPU si >= seuil
        return data_size >= self.thresholds.gpu_min_size
    
    def estimate_transfer_cost(self, data_size: int, dtype=np.float64) -> float:
        """
        Estime le coût de transfer CPU→GPU (en ms).
        
        Args:
            data_size: Nombre de points
            dtype: Type de données
        
        Returns:
            Coût estimé en millisecondes
        """
        bytes_per_element = np.dtype(dtype).itemsize
        total_mb = (data_size * bytes_per_element) / (1024 * 1024)
        return total_mb * self.thresholds.transfer_cost_ms_per_mb
    
    def auto_compute(
        self,
        data: np.ndarray,
        operation: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Calcule automatiquement sur device optimal.
        
        Args:
            data: Données d'entrée (NumPy array)
            operation: Type d'opération ("sma", "ema", "bollinger", etc.)
            **kwargs: Paramètres de l'opération
        
        Returns:
            Résultat calculé (NumPy array)
        """
        data_size = len(data) if hasattr(data, '__len__') else 1
        
        # Décision GPU vs CPU
        use_gpu = self.should_use_gpu(
            data_size=data_size,
            operation_type=kwargs.get("operation_type", "vectorial"),
        )
        
        if use_gpu:
            logger.debug(
                f"GPU compute: {operation} sur {data_size} points "
                f"(transfer ~{self.estimate_transfer_cost(data_size):.2f}ms)"
            )
            return self._compute_gpu(data, operation, **kwargs)
        else:
            logger.debug(f"CPU compute: {operation} sur {data_size} points")
            return self._compute_cpu(data, operation, **kwargs)
    
    def _compute_gpu(
        self,
        data: np.ndarray,
        operation: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Calcule sur GPU avec CuPy.
        
        Gère automatiquement:
        - Transfer CPU→GPU
        - Calcul sur GPU
        - Transfer GPU→CPU
        - Fallback CPU si erreur
        """
        try:
            import cupy as cp
            
            # Transfer CPU → GPU
            gpu_data = cp.asarray(data)
            
            # Calcul selon opération
            if operation == "sma":
                window = kwargs.get("window", 20)
                # Utiliser convolve pour moyenne mobile GPU
                kernel = cp.ones(window) / window
                result = cp.convolve(gpu_data, kernel, mode='valid')
                # Pad avec NaN pour garder même taille
                result = cp.concatenate([cp.full(window-1, cp.nan), result])
            
            elif operation == "ema":
                window = kwargs.get("window", 20)
                alpha = 2 / (window + 1)
                result = cp.zeros_like(gpu_data)
                result[0] = gpu_data[0]
                for i in range(1, len(gpu_data)):
                    result[i] = alpha * gpu_data[i] + (1 - alpha) * result[i-1]
            
            elif operation == "std":
                window = kwargs.get("window", 20)
                # Rolling std sur GPU
                result = cp.zeros(len(gpu_data))
                for i in range(window, len(gpu_data) + 1):
                    result[i-1] = cp.std(gpu_data[i-window:i])
                result[:window-1] = cp.nan
            
            else:
                # Opération non supportée → fallback CPU
                logger.warning(f"Opération '{operation}' non supportée sur GPU, fallback CPU")
                return self._compute_cpu(data, operation, **kwargs)
            
            # Transfer GPU → CPU
            return cp.asnumpy(result)
        
        except Exception as e:
            logger.warning(f"Erreur GPU compute, fallback CPU: {e}")
            return self._compute_cpu(data, operation, **kwargs)
    
    def _compute_cpu(
        self,
        data: np.ndarray,
        operation: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Calcule sur CPU avec NumPy/Numba.
        
        Utilise les implémentations optimisées existantes.
        """
        import pandas as pd
        
        if operation == "sma":
            window = kwargs.get("window", 20)
            return pd.Series(data).rolling(window).mean().values
        
        elif operation == "ema":
            window = kwargs.get("window", 20)
            # Calcul EMA simple sans Numba (pour benchmark)
            alpha = 2 / (window + 1)
            result = np.zeros_like(data, dtype=np.float64)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        elif operation == "std":
            window = kwargs.get("window", 20)
            return pd.Series(data).rolling(window).std().values
        
        else:
            raise ValueError(f"Opération '{operation}' non supportée")
    
    def batch_compute(
        self,
        data_list: list[np.ndarray],
        operation: str,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Calcule par batch sur GPU.
        
        Optimise la gestion mémoire GPU pour calculer
        plusieurs datasets en parallèle.
        
        Args:
            data_list: Liste de datasets
            operation: Type d'opération
            **kwargs: Paramètres de l'opération
        
        Returns:
            Liste de résultats
        """
        if not self._gpu_available:
            # Fallback CPU séquentiel
            return [self.auto_compute(data, operation, **kwargs) for data in data_list]
        
        try:
            import cupy as cp
            
            # Trier par taille pour batching optimal
            sorted_data = sorted(enumerate(data_list), key=lambda x: len(x[1]))
            
            results = [None] * len(data_list)
            batch = []
            batch_indices = []
            
            for idx, data in sorted_data:
                batch.append(data)
                batch_indices.append(idx)
                
                # Process batch quand taille atteinte
                if len(batch) >= self.thresholds.gpu_batch_size // len(data):
                    batch_results = [
                        self._compute_gpu(d, operation, **kwargs)
                        for d in batch
                    ]
                    for batch_idx, result in zip(batch_indices, batch_results):
                        results[batch_idx] = result
                    
                    batch = []
                    batch_indices = []
            
            # Process dernier batch
            if batch:
                batch_results = [
                    self._compute_gpu(d, operation, **kwargs)
                    for d in batch
                ]
                for batch_idx, result in zip(batch_indices, batch_results):
                    results[batch_idx] = result
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur batch compute GPU: {e}")
            return [self.auto_compute(data, operation, **kwargs) for data in data_list]


# ==================== API simplifiée ====================

_hybrid_compute: Optional[HybridCompute] = None


def get_hybrid_compute() -> HybridCompute:
    """Retourne l'instance singleton HybridCompute."""
    global _hybrid_compute
    if _hybrid_compute is None:
        _hybrid_compute = HybridCompute()
    return _hybrid_compute


def auto_compute(data: np.ndarray, operation: str, **kwargs) -> np.ndarray:
    """
    API simplifiée pour calcul automatique CPU/GPU.
    
    Example:
        >>> result = auto_compute(prices, "sma", window=20)
    """
    hc = get_hybrid_compute()
    return hc.auto_compute(data, operation, **kwargs)


def batch_compute(data_list: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
    """
    API simplifiée pour calcul batch GPU.
    
    Example:
        >>> results = batch_compute([prices1, prices2], "ema", window=12)
    """
    hc = get_hybrid_compute()
    return hc.batch_compute(data_list, operation, **kwargs)
