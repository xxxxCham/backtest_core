"""
Module-ID: utils.gpu_oom

Purpose: GPU OOM Handler - fallback automatique CPU si mémoire GPU insuffisante.

Role in pipeline: performance / resilience

Key components: GPUOOMHandler, gpu_memory_available, fallback_to_cpu context manager

Inputs: Mémoire estimée requise, device

Outputs: Calculs GPU ou fallback CPU, notifications

Dependencies: cupy (optionnel), numpy, logging, contextlib

Conventions: Estimation proactive mémoire; nettoyage cache GPU avant fallback; transparent pour caller.

Read-if: Modification estimation mémoire, logique fallback.

Skip-if: Vous utilisez juste @gpu_safe decorator.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MemoryStatus(Enum):
    """État de la mémoire GPU."""
    OK = "ok"                    # Mémoire suffisante
    LOW = "low"                  # Mémoire basse
    CRITICAL = "critical"        # Mémoire critique
    OOM = "oom"                  # Out of memory
    UNAVAILABLE = "unavailable"  # GPU non disponible


@dataclass
class GPUMemoryInfo:
    """Informations sur la mémoire GPU."""
    total: int = 0           # Mémoire totale en bytes
    used: int = 0            # Mémoire utilisée
    free: int = 0            # Mémoire libre
    status: MemoryStatus = MemoryStatus.UNAVAILABLE

    @property
    def usage_percent(self) -> float:
        """Pourcentage de mémoire utilisée."""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100

    @property
    def free_mb(self) -> float:
        """Mémoire libre en MB."""
        return self.free / (1024 * 1024)

    @property
    def free_gb(self) -> float:
        """Mémoire libre en GB."""
        return self.free / (1024 ** 3)


class GPUOOMHandler:
    """
    Gestionnaire d'erreurs Out-Of-Memory GPU.

    Surveille la mémoire GPU et gère les fallbacks vers CPU.

    Example:
        >>> handler = GPUOOMHandler()
        >>>
        >>> @handler.safe_gpu_operation
        >>> def compute_on_gpu(data):
        >>>     import cupy as cp
        >>>     return cp.sum(cp.array(data))
        >>>
        >>> # Tente sur GPU, fallback sur CPU si OOM
        >>> result = compute_on_gpu(large_array)
    """

    def __init__(
        self,
        low_memory_threshold: float = 0.2,    # 20% libre
        critical_threshold: float = 0.1,       # 10% libre
        auto_clear_on_low: bool = True,
        fallback_to_cpu: bool = True,
    ):
        """
        Args:
            low_memory_threshold: Seuil mémoire basse (fraction libre)
            critical_threshold: Seuil critique (fraction libre)
            auto_clear_on_low: Nettoyage auto si mémoire basse
            fallback_to_cpu: Fallback automatique vers CPU si OOM
        """
        self.low_threshold = low_memory_threshold
        self.critical_threshold = critical_threshold
        self.auto_clear = auto_clear_on_low
        self.fallback_to_cpu = fallback_to_cpu

        self._gpu_available = self._check_gpu_available()
        self._oom_count = 0
        self._fallback_count = 0

    def _check_gpu_available(self) -> bool:
        """Vérifie si CuPy/GPU est disponible."""
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            return True
        except (ImportError, Exception):
            return False

    def get_memory_info(self) -> GPUMemoryInfo:
        """
        Récupère les informations mémoire GPU.

        Returns:
            GPUMemoryInfo avec état actuel
        """
        if not self._gpu_available:
            return GPUMemoryInfo(status=MemoryStatus.UNAVAILABLE)

        try:
            import cupy as cp

            mem = cp.cuda.Device(0).mem_info
            free, total = mem[0], mem[1]
            used = total - free

            # Déterminer le status
            free_ratio = free / total if total > 0 else 0

            if free_ratio <= self.critical_threshold:
                status = MemoryStatus.CRITICAL
            elif free_ratio <= self.low_threshold:
                status = MemoryStatus.LOW
            else:
                status = MemoryStatus.OK

            return GPUMemoryInfo(
                total=total,
                used=used,
                free=free,
                status=status,
            )

        except Exception as e:
            logger.warning(f"Impossible de lire la mémoire GPU: {e}")
            return GPUMemoryInfo(status=MemoryStatus.UNAVAILABLE)

    def clear_memory(self) -> bool:
        """
        Libère la mémoire GPU.

        Returns:
            True si nettoyage réussi
        """
        if not self._gpu_available:
            return False

        try:
            import cupy as cp
            import gc

            # Garbage collection Python
            gc.collect()

            # Libérer les pools CuPy
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            # Synchroniser
            cp.cuda.Stream.null.synchronize()

            logger.debug("Mémoire GPU libérée")
            return True

        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage mémoire: {e}")
            return False

    def estimate_memory_required(self, data_shape: Tuple[int, ...], dtype=np.float64) -> int:
        """
        Estime la mémoire requise pour des données.

        Args:
            data_shape: Shape des données
            dtype: Type de données

        Returns:
            Mémoire estimée en bytes
        """
        element_size = np.dtype(dtype).itemsize
        num_elements = np.prod(data_shape)

        # Ajouter 20% pour overhead CuPy
        return int(num_elements * element_size * 1.2)

    def can_allocate(self, size_bytes: int) -> bool:
        """
        Vérifie si une allocation est possible.

        Args:
            size_bytes: Taille à allouer en bytes

        Returns:
            True si allocation possible
        """
        mem_info = self.get_memory_info()

        if mem_info.status == MemoryStatus.UNAVAILABLE:
            return False

        # Garder une marge de sécurité de 10%
        available = mem_info.free * 0.9
        return size_bytes <= available

    def check_and_prepare(self, required_bytes: int) -> bool:
        """
        Vérifie la mémoire et prépare si nécessaire.

        Args:
            required_bytes: Mémoire requise

        Returns:
            True si prêt pour l'allocation
        """
        mem_info = self.get_memory_info()

        # GPU non disponible
        if mem_info.status == MemoryStatus.UNAVAILABLE:
            return False

        # Vérifier si assez de mémoire
        if self.can_allocate(required_bytes):
            return True

        # Tenter de libérer la mémoire
        if self.auto_clear:
            self.clear_memory()

            # Re-vérifier
            if self.can_allocate(required_bytes):
                return True

        return False

    def handle_oom(self, exc: Exception) -> bool:
        """
        Gère une erreur OOM.

        Args:
            exc: Exception OOM capturée

        Returns:
            True si récupération tentée
        """
        self._oom_count += 1
        logger.warning(f"GPU OOM détecté (count={self._oom_count}): {exc}")

        # Tenter le nettoyage
        self.clear_memory()

        return True

    def safe_gpu_operation(self, func: Callable) -> Callable:
        """
        Décorateur pour opérations GPU sécurisées.

        Attrape les OOM et fallback vers CPU si configuré.

        Example:
            >>> @handler.safe_gpu_operation
            >>> def gpu_compute(data):
            >>>     import cupy as cp
            >>>     return cp.sum(cp.array(data))
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Vérifier mémoire avant
            mem_info = self.get_memory_info()

            if mem_info.status == MemoryStatus.CRITICAL and self.auto_clear:
                self.clear_memory()

            try:
                return func(*args, **kwargs)

            except Exception as exc:
                # Vérifier si c'est une erreur OOM
                error_msg = str(exc).lower()
                is_oom = any(w in error_msg for w in [
                    "out of memory", "oom", "memory allocation",
                    "cuda", "gpu memory", "cupy"
                ])

                if is_oom or isinstance(exc, MemoryError):
                    self.handle_oom(exc)

                    if self.fallback_to_cpu:
                        self._fallback_count += 1
                        logger.info("Fallback vers CPU...")

                        # Retenter avec numpy
                        return self._run_on_cpu(func, *args, **kwargs)

                raise

        return wrapper

    def _run_on_cpu(self, func: Callable, *args, **kwargs) -> Any:
        """
        Exécute une fonction en forçant CPU.

        Convertit les arrays CuPy en NumPy.
        """
        try:
            import cupy as cp

            # Convertir les args cupy -> numpy
            new_args = []
            for arg in args:
                if isinstance(arg, cp.ndarray):
                    new_args.append(cp.asnumpy(arg))
                else:
                    new_args.append(arg)

            # Convertir les kwargs
            new_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, cp.ndarray):
                    new_kwargs[key] = cp.asnumpy(val)
                else:
                    new_kwargs[key] = val

            return func(*new_args, **new_kwargs)

        except ImportError:
            # CuPy pas installé, exécuter directement
            return func(*args, **kwargs)

    @contextmanager
    def memory_guard(self, required_mb: float = 100):
        """
        Context manager pour opérations nécessitant de la mémoire GPU.

        Args:
            required_mb: Mémoire requise en MB

        Example:
            >>> with handler.memory_guard(required_mb=500):
            >>>     result = heavy_gpu_computation()
        """
        required_bytes = int(required_mb * 1024 * 1024)

        # Vérifier et préparer
        if not self.check_and_prepare(required_bytes):
            if self.fallback_to_cpu:
                logger.warning(
                    f"Mémoire GPU insuffisante ({required_mb}MB requis), "
                    "utilisation CPU recommandée"
                )
            else:
                raise MemoryError(
                    f"Mémoire GPU insuffisante: {required_mb}MB requis, "
                    f"{self.get_memory_info().free_mb:.0f}MB disponible"
                )

        try:
            yield
        finally:
            # Nettoyage optionnel après l'opération
            mem_info = self.get_memory_info()
            if mem_info.status in (MemoryStatus.LOW, MemoryStatus.CRITICAL):
                self.clear_memory()

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du handler."""
        mem_info = self.get_memory_info()

        return {
            "gpu_available": self._gpu_available,
            "oom_count": self._oom_count,
            "fallback_count": self._fallback_count,
            "memory_status": mem_info.status.value,
            "memory_free_mb": mem_info.free_mb,
            "memory_usage_percent": mem_info.usage_percent,
        }


# Singleton global
_oom_handler: Optional[GPUOOMHandler] = None


def get_oom_handler() -> GPUOOMHandler:
    """Retourne le handler OOM singleton."""
    global _oom_handler
    if _oom_handler is None:
        _oom_handler = GPUOOMHandler()
    return _oom_handler


def safe_gpu(func: Callable) -> Callable:
    """
    Décorateur raccourci pour opérations GPU sécurisées.

    Example:
        >>> @safe_gpu
        >>> def my_gpu_function(data):
        >>>     import cupy as cp
        >>>     return cp.sum(cp.array(data))
    """
    return get_oom_handler().safe_gpu_operation(func)


@contextmanager
def gpu_memory_guard(required_mb: float = 100):
    """Context manager raccourci pour garde mémoire GPU."""
    with get_oom_handler().memory_guard(required_mb):
        yield


def clear_gpu_memory():
    """Libère la mémoire GPU."""
    return get_oom_handler().clear_memory()


def get_gpu_memory_status() -> GPUMemoryInfo:
    """Retourne le status mémoire GPU."""
    return get_oom_handler().get_memory_info()


__all__ = [
    "MemoryStatus",
    "GPUMemoryInfo",
    "GPUOOMHandler",
    "get_oom_handler",
    "safe_gpu",
    "gpu_memory_guard",
    "clear_gpu_memory",
    "get_gpu_memory_status",
]
