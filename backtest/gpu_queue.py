"""
Module-ID: backtest.gpu_queue

Purpose: File d'attente GPU centralisée avec batching intelligent pour calcul indicateurs.

Architecture:
- GPUWorkerProcess : Processus dédié qui écoute la queue et traite les requêtes par batch
- Batching intelligent : Groupe les requêtes similaires pour maximiser throughput GPU
- Cache integration : Vérifie IndicatorBank avant calcul

Role in pipeline: performance optimization (GPU parallelization)

Key components:
- GPURequest/GPUResponse : Data structures pour communication inter-process
- GPUBatchProcessor : Logique de batching intelligent
- GPUWorkerProcess : Processus GPU dédié avec event loop

Inputs: Requêtes indicateurs (name, params, data_hash) via multiprocessing.Queue

Outputs: Résultats indicateurs calculés sur GPU

Dependencies: multiprocessing, cupy, performance.gpu, data.indicator_bank

Conventions:
- Batch size optimal : 10-50 requêtes (trade-off latence vs throughput)
- Timeout batch : 50ms max (évite latence excessive si peu de requêtes)
- Cache check avant GPU : Évite calculs redondants

Read-if: Modification logique batching ou GPU worker process.

Skip-if: Vous utilisez juste la queue (via calculate_indicator_gpu()).
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================== Request/Response Data Structures ========================


class RequestStatus(Enum):
    """Statut d'une requête GPU."""
    PENDING = "pending"
    BATCHED = "batched"
    COMPUTING = "computing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPURequest:
    """
    Requête de calcul d'indicateur sur GPU.

    Attributes:
        request_id: ID unique de la requête
        indicator_name: Nom de l'indicateur (bollinger, atr, rsi, etc.)
        params: Paramètres de l'indicateur
        data_hash: Hash des données OHLCV (pour cache lookup)
        df_pickle: DataFrame sérialisé (pickle bytes) - None si cache hit
        timestamp: Timestamp de création de la requête
        worker_id: ID du worker qui a envoyé la requête (pour debug)
    """
    request_id: str
    indicator_name: str
    params: Dict[str, Any]
    data_hash: str
    df_pickle: Optional[bytes] = None  # None si en cache
    timestamp: float = field(default_factory=time.time)
    worker_id: Optional[int] = None

    def __hash__(self):
        """Hash pour grouping dans batch."""
        params_str = str(sorted(self.params.items()))
        return hash((self.indicator_name, params_str))

    def batch_key(self) -> Tuple[str, str]:
        """Clé pour regrouper requêtes similaires dans un batch."""
        params_str = str(sorted(self.params.items()))
        return (self.indicator_name, params_str)


@dataclass
class GPUResponse:
    """
    Réponse d'une requête GPU.

    Attributes:
        request_id: ID de la requête correspondante
        result: Résultat du calcul (numpy array ou tuple d'arrays)
        success: True si succès, False si erreur
        error: Message d'erreur si échec
        compute_time_ms: Temps de calcul GPU en millisecondes
        was_cached: True si résultat vient du cache (pas de calcul GPU)
    """
    request_id: str
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    compute_time_ms: float = 0.0
    was_cached: bool = False


# ======================== Batching Logic ========================


class GPUBatchProcessor:
    """
    Processeur de batch pour calculs GPU optimisés.

    Stratégie de batching :
    1. Collecter requêtes pendant max_wait_ms (défaut: 50ms)
    2. Grouper par (indicator_name, params) → batch keys
    3. Pour chaque batch key, calculer en parallèle sur GPU
    4. Dispatcher les résultats aux workers correspondants

    Gains attendus :
    - 10 requêtes similaires → ~5-8x speedup vs séquentiel
    - 50 requêtes similaires → ~15-30x speedup vs séquentiel
    """

    def __init__(
        self,
        max_batch_size: int = 50,
        max_wait_ms: float = 50.0,
        use_cache: bool = True
    ):
        """
        Initialise le batch processor.

        Args:
            max_batch_size: Taille max d'un batch (défaut: 50)
            max_wait_ms: Temps max d'attente pour remplir un batch (défaut: 50ms)
            use_cache: Utiliser IndicatorBank pour cache (défaut: True)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.use_cache = use_cache

        # GPU calculator (lazy init)
        self._gpu_calc = None
        self._indicator_bank = None

        # Stats basiques
        self.total_requests = 0
        self.cache_hits = 0
        self.gpu_computes = 0
        self.total_batch_count = 0

        # ═══════════════════════════════════════════════════════════════════════════
        # MONITORING AVANCÉ - Latence, throughput, distribution batch sizes
        # ═══════════════════════════════════════════════════════════════════════════
        self.batch_times_ms = []               # Historique des temps de batch
        self.batch_sizes = []                  # Historique des tailles de batch
        self.compute_times_ms = []             # Temps de calcul GPU (sans cache)
        self.total_compute_time_ms = 0.0       # Temps total GPU compute
        self.total_batch_time_ms = 0.0         # Temps total batches (avec cache)

        # Compteurs par indicateur
        self.requests_per_indicator = {}       # {indicator_name: count}
        self.computes_per_indicator = {}       # {indicator_name: count}
        self.cache_hits_per_indicator = {}     # {indicator_name: count}

    def _init_gpu(self):
        """Initialisation lazy du GPU calculator."""
        if self._gpu_calc is not None:
            return

        try:
            from performance.gpu import GPUIndicatorCalculator, gpu_available

            if not gpu_available():
                raise RuntimeError("GPU non disponible")

            # Force GPU usage (pas de seuil min_samples)
            self._gpu_calc = GPUIndicatorCalculator(use_gpu=True, min_samples=0)
            logger.info("GPUBatchProcessor: GPU calculator initialisé")

        except Exception as e:
            logger.error(f"GPUBatchProcessor: Erreur init GPU - {e}")
            raise

    def _init_cache(self):
        """Initialisation lazy de l'IndicatorBank."""
        if not self.use_cache or self._indicator_bank is not None:
            return

        try:
            from data.indicator_bank import get_indicator_bank
            self._indicator_bank = get_indicator_bank()
            logger.info("GPUBatchProcessor: IndicatorBank initialisé")
        except Exception as e:
            logger.warning(f"GPUBatchProcessor: Cache désactivé - {e}")
            self.use_cache = False

    def process_batch(self, requests: List[GPURequest]) -> List[GPUResponse]:
        """
        Traite un batch de requêtes sur GPU.

        Args:
            requests: Liste de requêtes à traiter

        Returns:
            Liste de réponses (même ordre que requests)
        """
        if not requests:
            return []

        self._init_gpu()
        if self.use_cache:
            self._init_cache()

        self.total_requests += len(requests)
        self.total_batch_count += 1

        responses = []
        start_time = time.time()

        # Grouper par batch key (indicator + params)
        batches: Dict[Tuple[str, str], List[GPURequest]] = defaultdict(list)
        for req in requests:
            batches[req.batch_key()].append(req)

        logger.debug(
            f"GPUBatchProcessor: Traitement de {len(requests)} requêtes "
            f"en {len(batches)} batch(es) GPU"
        )

        # Traiter chaque batch
        for batch_key, batch_requests in batches.items():
            indicator_name, _ = batch_key

            for req in batch_requests:
                # 1. Check cache d'abord
                if self._indicator_bank is not None:
                    try:
                        # Deserialize DataFrame si besoin
                        df = pickle.loads(req.df_pickle) if req.df_pickle else None

                        if df is not None:
                            cached_result = self._indicator_bank.get(
                                req.indicator_name,
                                req.params,
                                df,
                                data_hash=req.data_hash,
                                backend="gpu"
                            )

                            if cached_result is not None:
                                self.cache_hits += 1
                                # Enregistrer cache hit par indicateur
                                ind_name = req.indicator_name
                                self.cache_hits_per_indicator[ind_name] = self.cache_hits_per_indicator.get(ind_name, 0) + 1

                                responses.append(GPUResponse(
                                    request_id=req.request_id,
                                    result=cached_result,
                                    success=True,
                                    compute_time_ms=0.0,
                                    was_cached=True
                                ))
                                continue
                    except Exception as e:
                        logger.debug(f"Cache lookup failed: {e}")

                # 2. Calculer sur GPU
                try:
                    compute_start = time.time()

                    # Deserialize DataFrame
                    df = pickle.loads(req.df_pickle)

                    # Calculer selon type d'indicateur
                    result = self._compute_indicator(
                        req.indicator_name,
                        df,
                        req.params
                    )

                    compute_time_ms = (time.time() - compute_start) * 1000
                    self.gpu_computes += 1

                    # Enregistrer métriques GPU compute
                    self.compute_times_ms.append(compute_time_ms)
                    self.total_compute_time_ms += compute_time_ms
                    ind_name = req.indicator_name
                    self.computes_per_indicator[ind_name] = self.computes_per_indicator.get(ind_name, 0) + 1

                    # 3. Mettre en cache
                    if self._indicator_bank is not None:
                        try:
                            self._indicator_bank.put(
                                req.indicator_name,
                                req.params,
                                df,
                                result,
                                data_hash=req.data_hash,
                                backend="gpu"
                            )
                        except Exception as e:
                            logger.debug(f"Cache put failed: {e}")

                    responses.append(GPUResponse(
                        request_id=req.request_id,
                        result=result,
                        success=True,
                        compute_time_ms=compute_time_ms,
                        was_cached=False
                    ))

                except Exception as e:
                    logger.error(f"GPUBatchProcessor: Erreur calcul {indicator_name} - {e}")
                    responses.append(GPUResponse(
                        request_id=req.request_id,
                        success=False,
                        error=str(e)
                    ))

        batch_time_ms = (time.time() - start_time) * 1000
        cache_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        # ═══════════════════════════════════════════════════════════════════════════
        # MONITORING - Enregistrer métriques du batch
        # ═══════════════════════════════════════════════════════════════════════════
        self.batch_times_ms.append(batch_time_ms)
        self.batch_sizes.append(len(requests))
        self.total_batch_time_ms += batch_time_ms

        # Compteurs par indicateur
        for req in requests:
            ind_name = req.indicator_name
            self.requests_per_indicator[ind_name] = self.requests_per_indicator.get(ind_name, 0) + 1

        logger.info(
            f"GPUBatchProcessor: Batch #{self.total_batch_count} traité - "
            f"{len(requests)} requêtes en {batch_time_ms:.1f}ms "
            f"(cache rate: {cache_rate:.1f}%)"
        )

        return responses

    def _compute_indicator(
        self,
        indicator_name: str,
        df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Any:
        """
        Calcule un indicateur sur GPU.

        Args:
            indicator_name: Nom de l'indicateur
            df: DataFrame OHLCV
            params: Paramètres

        Returns:
            Résultat du calcul (array ou tuple)
        """
        if self._gpu_calc is None:
            raise RuntimeError("GPU calculator non initialisé")

        # Dispatcher selon type d'indicateur
        if indicator_name == "bollinger":
            period = int(params.get("period", 20))
            std_dev = float(params.get("std_dev", 2.0))
            return self._gpu_calc.bollinger_bands(df["close"], period=period, std_dev=std_dev)

        elif indicator_name == "atr":
            period = int(params.get("period", 14))
            return self._gpu_calc.atr(df["high"], df["low"], df["close"], period=period)

        elif indicator_name == "rsi":
            period = int(params.get("period", 14))
            return self._gpu_calc.rsi(df["close"], period=period)

        elif indicator_name == "ema":
            period = int(params.get("period", 20))
            return self._gpu_calc.ema(df["close"], period=period)

        elif indicator_name == "sma":
            period = int(params.get("period", 20))
            return self._gpu_calc.sma(df["close"], period=period)

        elif indicator_name == "macd":
            fast_period = int(params.get("fast_period", 12))
            slow_period = int(params.get("slow_period", 26))
            signal_period = int(params.get("signal_period", 9))
            macd_line, signal_line, histogram = self._gpu_calc.macd(
                df["close"],
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period
            )
            return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

        else:
            raise ValueError(f"Indicateur non supporté: {indicator_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées du batch processor.

        Returns:
            Dict avec métriques : throughput, latence, cache rate, distribution batch sizes, etc.
        """
        cache_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        # Calcul des percentiles de latence batch
        import numpy as np
        batch_times_arr = np.array(self.batch_times_ms) if self.batch_times_ms else np.array([0])
        batch_sizes_arr = np.array(self.batch_sizes) if self.batch_sizes else np.array([0])
        compute_times_arr = np.array(self.compute_times_ms) if self.compute_times_ms else np.array([0])

        stats = {
            # Stats globales
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "gpu_computes": self.gpu_computes,
            "cache_hit_rate_pct": cache_rate,
            "total_batches": self.total_batch_count,

            # Throughput
            "avg_batch_size": self.total_requests / self.total_batch_count if self.total_batch_count > 0 else 0,
            "requests_per_second": (self.total_requests / (self.total_batch_time_ms / 1000)) if self.total_batch_time_ms > 0 else 0,

            # Latence batch (avec cache)
            "batch_latency_avg_ms": float(np.mean(batch_times_arr)) if len(batch_times_arr) > 0 else 0,
            "batch_latency_p50_ms": float(np.percentile(batch_times_arr, 50)) if len(batch_times_arr) > 0 else 0,
            "batch_latency_p95_ms": float(np.percentile(batch_times_arr, 95)) if len(batch_times_arr) > 0 else 0,
            "batch_latency_p99_ms": float(np.percentile(batch_times_arr, 99)) if len(batch_times_arr) > 0 else 0,
            "batch_latency_max_ms": float(np.max(batch_times_arr)) if len(batch_times_arr) > 0 else 0,

            # Latence GPU compute (sans cache)
            "gpu_compute_avg_ms": float(np.mean(compute_times_arr)) if len(compute_times_arr) > 0 else 0,
            "gpu_compute_p95_ms": float(np.percentile(compute_times_arr, 95)) if len(compute_times_arr) > 0 else 0,
            "total_gpu_compute_time_ms": self.total_compute_time_ms,

            # Distribution batch sizes
            "batch_size_min": int(np.min(batch_sizes_arr)) if len(batch_sizes_arr) > 0 else 0,
            "batch_size_max": int(np.max(batch_sizes_arr)) if len(batch_sizes_arr) > 0 else 0,
            "batch_size_avg": float(np.mean(batch_sizes_arr)) if len(batch_sizes_arr) > 0 else 0,
            "batch_size_p50": float(np.percentile(batch_sizes_arr, 50)) if len(batch_sizes_arr) > 0 else 0,

            # Stats par indicateur
            "requests_per_indicator": dict(self.requests_per_indicator),
            "computes_per_indicator": dict(self.computes_per_indicator),
            "cache_hits_per_indicator": dict(self.cache_hits_per_indicator),
        }

        return stats


# ======================== GPU Worker Process ========================


class GPUWorkerProcess:
    """
    Processus dédié qui écoute la queue de requêtes GPU et traite les batches.

    Event loop :
    1. Collecter requêtes pendant max_wait_ms
    2. Si batch plein OU timeout → traiter batch
    3. Dispatcher réponses via response queue
    4. Répéter jusqu'à signal STOP
    """

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        max_batch_size: int = 50,
        max_wait_ms: float = 50.0,
        use_cache: bool = True
    ):
        """
        Initialise le GPU worker process.

        Args:
            request_queue: Queue de requêtes (multiprocessing.Queue)
            response_queue: Queue de réponses (multiprocessing.Queue)
            max_batch_size: Taille max d'un batch
            max_wait_ms: Temps max d'attente pour remplir un batch
            use_cache: Utiliser IndicatorBank
        """
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.use_cache = use_cache

        self.processor = GPUBatchProcessor(
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            use_cache=use_cache
        )

        self._running = False
        self._process: Optional[mp.Process] = None

    def start(self):
        """Lance le processus GPU en background."""
        if self._process is not None and self._process.is_alive():
            logger.warning("GPUWorkerProcess déjà démarré")
            return

        self._running = True
        self._process = mp.Process(target=self._event_loop, daemon=True)
        self._process.start()
        logger.info(f"GPUWorkerProcess démarré (PID: {self._process.pid})")

    def stop(self, timeout: float = 5.0):
        """Arrête le processus GPU proprement."""
        if self._process is None or not self._process.is_alive():
            return

        logger.info("GPUWorkerProcess: Signal d'arrêt envoyé")

        # Envoyer signal STOP
        self.request_queue.put(("STOP", None))

        # Attendre fin du process
        self._process.join(timeout=timeout)

        if self._process.is_alive():
            logger.warning("GPUWorkerProcess: Timeout, terminaison forcée")
            self._process.terminate()
            self._process.join(timeout=1.0)

        logger.info("GPUWorkerProcess arrêté")

    def _event_loop(self):
        """
        Event loop principal du GPU worker.

        Collecte les requêtes et les traite par batch.
        """
        logger.info("GPUWorkerProcess: Event loop démarré")

        batch: List[GPURequest] = []
        last_batch_time = time.time()

        try:
            while self._running:
                # 1. Collecter requêtes (timeout = max_wait_ms)
                timeout_s = self.max_wait_ms / 1000.0

                try:
                    msg = self.request_queue.get(timeout=timeout_s)

                    # Check signal STOP
                    if isinstance(msg, tuple) and msg[0] == "STOP":
                        logger.info("GPUWorkerProcess: Signal STOP reçu")
                        break

                    # Ajouter au batch
                    if isinstance(msg, GPURequest):
                        batch.append(msg)

                except Exception:
                    # Timeout ou erreur → traiter le batch actuel
                    pass

                # 2. Décider si traiter le batch
                should_process = (
                    len(batch) >= self.max_batch_size or
                    (batch and (time.time() - last_batch_time) * 1000 >= self.max_wait_ms)
                )

                if should_process and batch:
                    # 3. Traiter le batch
                    responses = self.processor.process_batch(batch)

                    # 4. Dispatcher les réponses
                    for response in responses:
                        self.response_queue.put(response)

                    # Reset batch
                    batch = []
                    last_batch_time = time.time()

            # Traiter dernier batch si non vide
            if batch:
                logger.info(f"GPUWorkerProcess: Traitement du dernier batch ({len(batch)} requêtes)")
                responses = self.processor.process_batch(batch)
                for response in responses:
                    self.response_queue.put(response)

        except Exception as e:
            logger.error(f"GPUWorkerProcess: Erreur dans event loop - {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Log stats finales
            stats = self.processor.get_stats()
            logger.info(f"GPUWorkerProcess: Stats finales - {stats}")


# ======================== High-level API ========================


# Buffer local par worker pour stocker les réponses orphelines
# Évite le deadlock quand plusieurs workers lisent la response_queue
_response_buffer: Dict[str, GPUResponse] = {}


def calculate_indicator_gpu(
    indicator_name: str,
    df: pd.DataFrame,
    params: Dict[str, Any],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    timeout: float = 30.0
) -> Optional[Any]:
    """
    Calcule un indicateur via la GPU queue (API haut niveau).

    IMPORTANT: Utilise un buffer local pour éviter le deadlock quand
    plusieurs workers lisent simultanément la response_queue.

    Args:
        indicator_name: Nom de l'indicateur
        df: DataFrame OHLCV
        params: Paramètres
        request_queue: Queue de requêtes GPU
        response_queue: Queue de réponses GPU
        timeout: Timeout en secondes (défaut: 30s)

    Returns:
        Résultat du calcul ou None si timeout/erreur
    """
    global _response_buffer

    # Générer request_id unique (avec PID pour éviter collisions)
    request_id = hashlib.sha256(
        f"{time.time()}{os.getpid()}{id(df)}{id(params)}".encode()
    ).hexdigest()[:16]

    # Check buffer local d'abord (réponse reçue par un autre appel)
    if request_id in _response_buffer:
        response = _response_buffer.pop(request_id)
        if response.success:
            return response.result
        else:
            logger.error(f"GPU compute failed (cached): {response.error}")
            return None

    # Créer data_hash pour cache lookup
    from data.indicator_bank import IndicatorBank
    bank = IndicatorBank()
    data_hash = bank.get_data_hash(df)

    # Sérialiser DataFrame
    df_pickle = pickle.dumps(df)

    # Créer requête
    request = GPURequest(
        request_id=request_id,
        indicator_name=indicator_name,
        params=params,
        data_hash=data_hash,
        df_pickle=df_pickle,
        worker_id=os.getpid()
    )

    # Envoyer requête
    try:
        request_queue.put(request, timeout=5.0)
    except Exception as e:
        logger.error(f"GPU request queue full ou timeout: {e}")
        return None

    # Attendre réponse avec buffer local pour éviter deadlock
    start_time = time.time()
    check_count = 0

    while (time.time() - start_time) < timeout:
        try:
            # Check buffer local d'abord (réponse peut être arrivée entre temps)
            if request_id in _response_buffer:
                response = _response_buffer.pop(request_id)
                if response.success:
                    return response.result
                else:
                    logger.error(f"GPU compute failed: {response.error}")
                    return None

            # Lire une réponse de la queue (timeout court pour réactivité)
            response = response_queue.get(timeout=0.2)
            check_count += 1

            if response.request_id == request_id:
                # C'est notre réponse !
                if response.success:
                    return response.result
                else:
                    logger.error(f"GPU compute failed: {response.error}")
                    return None
            else:
                # Pas notre réponse → stocker dans buffer local
                # Ne PAS remettre en queue globale (évite deadlock)
                _response_buffer[response.request_id] = response

                # Nettoyer buffer si trop gros (évite fuite mémoire)
                if len(_response_buffer) > 100:
                    # Supprimer les plus vieilles entrées
                    oldest_keys = list(_response_buffer.keys())[:50]
                    for old_key in oldest_keys:
                        _response_buffer.pop(old_key, None)

        except Exception:
            # Timeout de get() ou autre erreur → continuer à attendre
            continue

    # Timeout final
    logger.error(
        f"GPU compute timeout après {timeout}s "
        f"(request_id={request_id}, checks={check_count}, buffer_size={len(_response_buffer)})"
    )
    return None


def format_gpu_stats(stats: Dict[str, Any]) -> str:
    """
    Formate joliment les stats GPU pour affichage.

    Args:
        stats: Dict retourné par GPUBatchProcessor.get_stats()

    Returns:
        String formaté avec stats GPU
    """
    if not stats.get("enabled", True):
        return "GPU Queue: Disabled"

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "GPU Queue Statistics",
        "═══════════════════════════════════════════════════════════════",
        "",
        "Throughput:",
        f"  Total requests:       {stats.get('total_requests', 0):,}",
        f"  GPU computes:         {stats.get('gpu_computes', 0):,}",
        f"  Cache hits:           {stats.get('cache_hits', 0):,} ({stats.get('cache_hit_rate_pct', 0):.1f}%)",
        f"  Total batches:        {stats.get('total_batches', 0):,}",
        f"  Avg batch size:       {stats.get('avg_batch_size', 0):.1f}",
        f"  Requests/sec:         {stats.get('requests_per_second', 0):.0f}",
        "",
        "Latency (Batch with cache):",
        f"  Average:              {stats.get('batch_latency_avg_ms', 0):.1f}ms",
        f"  P50:                  {stats.get('batch_latency_p50_ms', 0):.1f}ms",
        f"  P95:                  {stats.get('batch_latency_p95_ms', 0):.1f}ms",
        f"  P99:                  {stats.get('batch_latency_p99_ms', 0):.1f}ms",
        f"  Max:                  {stats.get('batch_latency_max_ms', 0):.1f}ms",
        "",
        "GPU Compute (without cache):",
        f"  Average:              {stats.get('gpu_compute_avg_ms', 0):.1f}ms",
        f"  P95:                  {stats.get('gpu_compute_p95_ms', 0):.1f}ms",
        f"  Total time:           {stats.get('total_gpu_compute_time_ms', 0) / 1000:.1f}s",
        "",
        "Batch Size Distribution:",
        f"  Min:                  {stats.get('batch_size_min', 0)}",
        f"  Avg:                  {stats.get('batch_size_avg', 0):.1f}",
        f"  P50:                  {stats.get('batch_size_p50', 0):.1f}",
        f"  Max:                  {stats.get('batch_size_max', 0)}",
    ]

    # Top 5 indicateurs
    requests_per_ind = stats.get("requests_per_indicator", {})
    if requests_per_ind:
        lines.extend([
            "",
            "Top 5 Indicators (by requests):",
        ])
        sorted_ind = sorted(requests_per_ind.items(), key=lambda x: x[1], reverse=True)[:5]
        for ind_name, count in sorted_ind:
            cache_hits = stats.get("cache_hits_per_indicator", {}).get(ind_name, 0)
            computes = stats.get("computes_per_indicator", {}).get(ind_name, 0)
            cache_rate = (cache_hits / count * 100) if count > 0 else 0
            lines.append(f"  {ind_name:12s}: {count:6,} req ({cache_rate:.1f}% cached, {computes:6,} GPU)")

    lines.append("═══════════════════════════════════════════════════════════════")

    return "\n".join(lines)


__all__ = [
    "GPURequest",
    "GPUResponse",
    "GPUBatchProcessor",
    "GPUWorkerProcess",
    "calculate_indicator_gpu",
    "format_gpu_stats",
]