"""
Module-ID: backtest.gpu_context

Purpose: Gestionnaire de contexte global pour GPU queue (singleton).

Architecture:
- GPUContextManager : Singleton qui gère le lifecycle des GPU queues
- Démarre GPU worker process au démarrage
- Arrête proprement le worker à la fin
- Fournit les queues aux workers CPU via configuration globale

Role in pipeline: GPU orchestration

Key components:
- GPUContextManager : Singleton pour gestion GPU queues
- init_gpu_context() : Initialise le contexte GPU (appelé avant sweep)
- cleanup_gpu_context() : Nettoie le contexte GPU (appelé après sweep)
- get_gpu_queues() : Retourne les queues configurées ou None

Dependencies: multiprocessing, backtest.gpu_queue

Conventions:
- Singleton pattern pour éviter multi-init
- Auto-cleanup via atexit si oublié
- Thread-safe pour accès concurrent

Read-if: Modification du lifecycle GPU ou configuration globale.

Skip-if: Vous utilisez simplement init_gpu_context() avant le sweep.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUContextManager:
    """
    Gestionnaire de contexte GPU (singleton).

    Gère le lifecycle complet des GPU queues :
    - Création des queues (multiprocessing.Queue)
    - Démarrage du GPU worker process
    - Arrêt propre du worker
    - Cleanup automatique via atexit

    Usage:
        >>> # Dans SweepEngine.run_sweep() ou avant sweep
        >>> init_gpu_context()
        >>>
        >>> # Les workers CPU récupèrent automatiquement les queues
        >>> # via indicators.registry.get_gpu_queues()
        >>>
        >>> # Après le sweep
        >>> cleanup_gpu_context()
    """

    _instance: Optional[GPUContextManager] = None
    _lock = mp.Lock()

    def __new__(cls):
        """Singleton pattern thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialise le gestionnaire (une seule fois)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._enabled = False
        self._request_queue: Optional[mp.Queue] = None
        self._response_queue: Optional[mp.Queue] = None
        self._gpu_worker: Optional[Any] = None
        self._max_batch_size = 50
        self._max_wait_ms = 50.0
        self._use_cache = True

        # Register cleanup
        atexit.register(self.cleanup)

    def init(
        self,
        max_batch_size: int = 50,
        max_wait_ms: float = 50.0,
        use_cache: bool = True,
        force: bool = False,
        indicator_cache_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialise le contexte GPU (queues + worker process).

        Args:
            max_batch_size: Taille max d'un batch GPU
            max_wait_ms: Temps max d'attente pour remplir un batch
            use_cache: Utiliser IndicatorBank pour cache
            force: Forcer réinitialisation si déjà initialisé
            indicator_cache_config: Configuration personnalisée pour IndicatorBank
                                   (memory_max_entries, disk_enabled, etc.)

        Returns:
            True si succès, False sinon
        """
        # Check si GPU queue activé
        enabled = os.getenv("BACKTEST_GPU_QUEUE_ENABLED", "1").strip() in ("1", "true", "yes", "on")
        if not enabled:
            logger.info("GPUContext: GPU queue désactivé (BACKTEST_GPU_QUEUE_ENABLED=0)")
            return False

        # Check si GPU disponible
        try:
            from performance.gpu import gpu_available
            if not gpu_available():
                logger.warning("GPUContext: GPU non disponible, désactivation GPU queue")
                return False
        except Exception as e:
            logger.warning(f"GPUContext: Erreur détection GPU - {e}")
            return False

        # Check si déjà initialisé
        if self._enabled and not force:
            logger.debug("GPUContext: Déjà initialisé")
            return True

        try:
            # Cleanup précédent si force
            if force and self._enabled:
                self.cleanup()

            logger.info("GPUContext: Initialisation des GPU queues...")

            # Configurer IndicatorBank avec config personnalisée si fournie
            if indicator_cache_config is not None:
                try:
                    from data.indicator_bank import get_indicator_bank
                    bank = get_indicator_bank(**indicator_cache_config)
                    logger.info(
                        f"IndicatorBank configuré: "
                        f"memory_entries={indicator_cache_config.get('memory_max_entries', 128)}, "
                        f"disk_enabled={indicator_cache_config.get('disk_enabled', True)}"
                    )
                except Exception as e:
                    logger.warning(f"Erreur configuration IndicatorBank: {e}")

            # Créer les queues (multiprocessing pour communication inter-process)
            self._request_queue = mp.Queue(maxsize=1000)
            self._response_queue = mp.Queue(maxsize=1000)
            self._max_batch_size = max_batch_size
            self._max_wait_ms = max_wait_ms
            self._use_cache = use_cache

            # Configurer les GPU queues dans le registry pour que calculate_indicator() les utilise
            from indicators.registry import set_gpu_queues
            set_gpu_queues(self._request_queue, self._response_queue)
            logger.info("GPU queues configurées dans indicators.registry")

            # Démarrer GPU worker process
            from backtest.gpu_queue import GPUWorkerProcess

            self._gpu_worker = GPUWorkerProcess(
                request_queue=self._request_queue,
                response_queue=self._response_queue,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
                use_cache=use_cache
            )

            self._gpu_worker.start()
            self._enabled = True

            logger.info(
                f"GPUContext: GPU worker démarré - "
                f"batch_size={max_batch_size}, wait_ms={max_wait_ms:.1f}, cache={use_cache}"
            )

            return True

        except Exception as e:
            logger.error(f"GPUContext: Erreur initialisation - {e}")
            self._enabled = False
            return False

    def cleanup(self):
        """Arrête proprement le GPU worker process."""
        if not self._enabled:
            return

        try:
            logger.info("GPUContext: Arrêt du GPU worker...")

            if self._gpu_worker is not None:
                self._gpu_worker.stop(timeout=5.0)
                self._gpu_worker = None

            if self._request_queue is not None:
                self._request_queue.close()
                self._request_queue = None

            if self._response_queue is not None:
                self._response_queue.close()
                self._response_queue = None

            self._enabled = False
            logger.info("GPUContext: Nettoyage terminé")

        except Exception as e:
            logger.error(f"GPUContext: Erreur cleanup - {e}")

    def get_queues(self) -> Optional[Tuple[mp.Queue, mp.Queue]]:
        """
        Retourne les GPU queues si initialisées.

        Returns:
            Tuple (request_queue, response_queue) ou None
        """
        if not self._enabled:
            return None

        if self._request_queue is None or self._response_queue is None:
            return None

        return (self._request_queue, self._response_queue)

    def is_enabled(self) -> bool:
        """Retourne True si GPU queue est active."""
        return self._enabled

    def get_stats(self) -> dict:
        """Retourne les stats du GPU worker."""
        if not self._enabled or self._gpu_worker is None:
            return {"enabled": False}

        try:
            stats = self._gpu_worker.processor.get_stats()
            stats["enabled"] = True
            return stats
        except Exception:
            return {"enabled": True, "error": "stats unavailable"}


# ======================== High-level API ========================

_gpu_context: Optional[GPUContextManager] = None


def get_gpu_context() -> GPUContextManager:
    """Retourne le contexte GPU singleton."""
    global _gpu_context
    if _gpu_context is None:
        _gpu_context = GPUContextManager()
    return _gpu_context


def init_gpu_context(
    max_batch_size: int = 50,
    max_wait_ms: float = 50.0,
    use_cache: bool = True,
    force: bool = False,
    indicator_cache_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Initialise le contexte GPU (queues + worker).

    À appeler avant le début d'un sweep pour activer GPU batching.

    Args:
        max_batch_size: Taille max d'un batch GPU (défaut: 50)
        max_wait_ms: Temps max d'attente pour remplir un batch (défaut: 50ms)
        use_cache: Utiliser IndicatorBank pour cache (défaut: True)
        force: Forcer réinitialisation si déjà initialisé
        indicator_cache_config: Configuration personnalisée pour IndicatorBank
                               (ex: {"memory_max_entries": 100000, "disk_enabled": False})

    Returns:
        True si succès, False sinon

    Example:
        >>> # Dans SweepEngine.run_sweep()
        >>> init_gpu_context(
        ...     max_batch_size=50,
        ...     max_wait_ms=50.0,
        ...     indicator_cache_config={"memory_max_entries": 100000, "disk_enabled": False}
        ... )
        >>> # ... exécuter le sweep ...
        >>> cleanup_gpu_context()
    """
    ctx = get_gpu_context()
    return ctx.init(
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        use_cache=use_cache,
        force=force,
        indicator_cache_config=indicator_cache_config
    )


def cleanup_gpu_context():
    """
    Nettoie le contexte GPU (arrête le worker).

    À appeler après le sweep pour libérer les ressources.
    """
    ctx = get_gpu_context()
    ctx.cleanup()


def get_gpu_queues() -> Optional[Tuple[mp.Queue, mp.Queue]]:
    """
    Retourne les GPU queues configurées.

    Returns:
        Tuple (request_queue, response_queue) ou None si non initialisé
    """
    ctx = get_gpu_context()
    return ctx.get_queues()


def is_gpu_context_enabled() -> bool:
    """Retourne True si GPU queue est active."""
    ctx = get_gpu_context()
    return ctx.is_enabled()


def get_gpu_context_stats() -> dict:
    """Retourne les stats du GPU context."""
    ctx = get_gpu_context()
    return ctx.get_stats()


__all__ = [
    "GPUContextManager",
    "init_gpu_context",
    "cleanup_gpu_context",
    "get_gpu_queues",
    "is_gpu_context_enabled",
    "get_gpu_context_stats",
]