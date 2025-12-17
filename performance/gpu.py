"""
Backtest Core - GPU Acceleration Module
=======================================

Accélération GPU optionnelle pour les calculs d'indicateurs.
Utilise CuPy et/ou Numba CUDA si disponibles.

Note: Ce module est optionnel. Si CuPy/Numba ne sont pas installés,
les calculs se feront sur CPU de manière transparente.

Installation GPU:
    pip install cupy-cuda12x  # Pour CUDA 12.x
    pip install numba

Usage:
    >>> from performance.gpu import gpu_available, GPUIndicatorCalculator
    >>>
    >>> if gpu_available():
    ...     calc = GPUIndicatorCalculator()
    ...     sma = calc.sma(prices, period=20)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================== Détection GPU ========================

# CuPy (GPU array operations)
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info(f"CuPy disponible: version {cp.__version__}")
except ImportError:
    HAS_CUPY = False
    cp = None

# Numba CUDA (JIT GPU kernels)
# NOTE: Désactivé car incompatible avec RTX 5080 (sm_90)
# Numba CUDA 0.61 ne supporte pas les architectures Blackwell.
# Utiliser CuPy à la place qui fonctionne correctement.
HAS_NUMBA_CUDA = False
cuda = None
float64 = None  # Pour éviter NameError si utilisé quelque part


# ======================== GPU Device Manager ========================

class GPUDeviceManager:
    """
    Gestionnaire de device GPU - Approche prudente single-GPU.
    
    Stratégie:
    1. Détecte tous les GPUs disponibles
    2. Sélectionne le plus puissant (par mémoire) par défaut
    3. Verrouille sur ce device pour toute la session
    4. Évite les switch de device intempestifs
    
    Environment variables:
        CUDA_VISIBLE_DEVICES: Limite les GPUs visibles
        BACKTEST_GPU_ID: Force un GPU spécifique (0, 1, ...)
    """
    
    _instance: Optional['GPUDeviceManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GPUDeviceManager._initialized:
            return
            
        self._device_id: Optional[int] = None
        self._device_name: str = "CPU"
        self._device_memory_gb: float = 0.0
        self._available_devices: List[dict] = []
        self._locked: bool = False
        
        if HAS_CUPY:
            self._detect_devices()
            self._select_best_device()
        
        GPUDeviceManager._initialized = True
    
    def _detect_devices(self) -> None:
        """Détecte tous les GPUs disponibles."""
        if not HAS_CUPY:
            return
            
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            logger.info(f"GPUDeviceManager: {device_count} GPU(s) détecté(s)")
            
            for device_id in range(device_count):
                try:
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                    
                    # Récupérer mémoire en activant temporairement le device
                    with cp.cuda.Device(device_id):
                        mem_info = cp.cuda.runtime.memGetInfo()
                        total_mem_gb = mem_info[1] / (1024**3)
                        free_mem_gb = mem_info[0] / (1024**3)
                    
                    device_info = {
                        "id": device_id,
                        "name": name,
                        "total_memory_gb": total_mem_gb,
                        "free_memory_gb": free_mem_gb,
                        "compute_capability": (props["major"], props["minor"]),
                    }
                    self._available_devices.append(device_info)
                    logger.info(f"  GPU {device_id}: {name} ({total_mem_gb:.1f} GB)")
                    
                except Exception as e:
                    logger.warning(f"  GPU {device_id}: Erreur détection - {e}")
                    
        except Exception as e:
            logger.error(f"GPUDeviceManager: Erreur énumération GPUs - {e}")
    
    def _select_best_device(self) -> None:
        """Sélectionne le meilleur GPU (plus de mémoire = plus puissant généralement)."""
        if not self._available_devices:
            logger.warning("GPUDeviceManager: Aucun GPU disponible")
            return
        
        # Vérifier si un GPU est forcé via variable d'environnement
        forced_gpu = os.environ.get("BACKTEST_GPU_ID")
        if forced_gpu is not None:
            try:
                forced_id = int(forced_gpu)
                matching = [d for d in self._available_devices if d["id"] == forced_id]
                if matching:
                    self._set_device(matching[0])
                    logger.info(f"GPUDeviceManager: GPU {forced_id} forcé via BACKTEST_GPU_ID")
                    return
                else:
                    logger.warning(f"GPUDeviceManager: GPU {forced_id} non trouvé, sélection auto")
            except ValueError:
                logger.warning(f"GPUDeviceManager: BACKTEST_GPU_ID invalide: {forced_gpu}")
        
        # Sélectionner le GPU avec le plus de mémoire totale
        best_device = max(self._available_devices, key=lambda d: d["total_memory_gb"])
        self._set_device(best_device)
        
        if len(self._available_devices) > 1:
            logger.info(
                f"GPUDeviceManager: Sélection automatique du GPU le plus puissant: "
                f"{best_device['name']} (GPU {best_device['id']})"
            )
    
    def _set_device(self, device_info: dict) -> None:
        """Configure et verrouille sur un device."""
        if not HAS_CUPY:
            return
            
        self._device_id = device_info["id"]
        self._device_name = device_info["name"]
        self._device_memory_gb = device_info["total_memory_gb"]
        
        # Activer le device et le verrouiller
        cp.cuda.Device(self._device_id).use()
        self._locked = True
        
        logger.info(
            f"GPUDeviceManager: Verrouillé sur GPU {self._device_id} "
            f"({self._device_name}, {self._device_memory_gb:.1f} GB)"
        )
    
    @property
    def device_id(self) -> Optional[int]:
        """ID du device actif."""
        return self._device_id
    
    @property
    def device_name(self) -> str:
        """Nom du device actif."""
        return self._device_name
    
    @property
    def available_devices(self) -> List[dict]:
        """Liste des devices disponibles."""
        return self._available_devices.copy()
    
    def ensure_device(self) -> None:
        """S'assure que le bon device est actif (appeler avant calculs GPU)."""
        if self._locked and self._device_id is not None and HAS_CUPY:
            current_device = cp.cuda.Device().id
            if current_device != self._device_id:
                logger.warning(
                    f"GPUDeviceManager: Device changé! "
                    f"Attendu {self._device_id}, actuel {current_device}. Correction..."
                )
                cp.cuda.Device(self._device_id).use()
    
    def get_info(self) -> dict:
        """Retourne les informations sur le GPU actif."""
        return {
            "device_id": self._device_id,
            "device_name": self._device_name,
            "device_memory_gb": self._device_memory_gb,
            "available_devices": len(self._available_devices),
            "locked": self._locked,
        }


# Singleton global - initialisé au premier accès
_gpu_manager: Optional[GPUDeviceManager] = None


def get_gpu_manager() -> GPUDeviceManager:
    """Retourne le gestionnaire GPU singleton."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUDeviceManager()
    return _gpu_manager


# Initialisation automatique au chargement du module si CuPy disponible
if HAS_CUPY:
    try:
        _gpu_manager = GPUDeviceManager()
    except Exception as e:
        logger.error(f"Erreur initialisation GPUDeviceManager: {e}")
        _gpu_manager = None


def gpu_available() -> bool:
    """Vérifie si le GPU est disponible."""
    return HAS_CUPY or HAS_NUMBA_CUDA


def get_gpu_info() -> dict:
    """Retourne les informations sur le GPU."""
    info = {
        "cupy_available": HAS_CUPY,
        "numba_cuda_available": HAS_NUMBA_CUDA,
        "gpu_available": gpu_available(),
    }
    
    if HAS_CUPY and _gpu_manager:
        manager_info = _gpu_manager.get_info()
        info.update({
            "cupy_device": manager_info["device_id"],
            "cupy_device_name": manager_info["device_name"],
            "cupy_memory_total_gb": manager_info["device_memory_gb"],
            "device_locked": manager_info["locked"],
            "available_gpu_count": manager_info["available_devices"],
        })
        
        # Ajouter mémoire libre actuelle
        if manager_info["device_id"] is not None:
            try:
                _gpu_manager.ensure_device()
                mem_info = cp.cuda.runtime.memGetInfo()
                info["cupy_memory_free_gb"] = mem_info[0] / (1024**3)
            except Exception as e:
                info["cupy_error"] = str(e)
    
    return info


# ======================== Array Abstraction ========================

def to_gpu(arr: np.ndarray) -> Any:
    """Transfère un array numpy vers le GPU."""
    if HAS_CUPY:
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """Transfère un array GPU vers le CPU."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def get_array_module(arr: Any):
    """Retourne le module array (numpy ou cupy) pour un array."""
    if HAS_CUPY:
        return cp.get_array_module(arr)
    return np


# ======================== GPU Indicator Calculator ========================

class GPUIndicatorCalculator:
    """
    Calculateur d'indicateurs avec accélération GPU.
    
    Utilise CuPy pour les opérations vectorielles sur GPU.
    Fallback automatique sur CPU si GPU non disponible.
    
    Le GPUDeviceManager garantit l'utilisation d'un seul GPU
    (le plus puissant par défaut) pour éviter les problèmes
    de switch entre GPUs.
    
    Example:
        >>> calc = GPUIndicatorCalculator(use_gpu=True)
        >>> 
        >>> # SMA sur GPU
        >>> sma = calc.sma(prices, period=20)
        >>> 
        >>> # EMA sur GPU
        >>> ema = calc.ema(prices, period=12)
        >>> 
        >>> # Bollinger Bands sur GPU
        >>> upper, middle, lower = calc.bollinger_bands(prices, period=20, std=2.0)
    """
    
    # Seuil minimum pour utiliser le GPU (overhead transfert)
    MIN_SAMPLES_FOR_GPU = 5000
    
    def __init__(self, use_gpu: bool = True, min_samples: int = 5000):
        """
        Initialise le calculateur GPU.
        
        Args:
            use_gpu: Activer le GPU si disponible
            min_samples: Seuil minimum pour utiliser le GPU
        """
        self.use_gpu = use_gpu and gpu_available()
        self.min_samples = min_samples
        self._gpu_manager = get_gpu_manager() if self.use_gpu else None
        
        if self.use_gpu and self._gpu_manager:
            info = self._gpu_manager.get_info()
            logger.info(
                f"GPUIndicatorCalculator: GPU activé - {info['device_name']} "
                f"(GPU {info['device_id']})"
            )
        else:
            logger.info("GPUIndicatorCalculator: Mode CPU")
    
    def _ensure_device(self) -> None:
        """S'assure que le bon GPU est actif avant calcul."""
        if self._gpu_manager:
            self._gpu_manager.ensure_device()
    
    def _should_use_gpu(self, n_samples: int) -> bool:
        """Détermine si le GPU doit être utilisé pour cette taille de données."""
        return self.use_gpu and n_samples >= self.min_samples
    
    def _to_array(self, data: Union[np.ndarray, pd.Series], use_gpu: bool) -> Any:
        """Convertit les données en array (GPU ou CPU)."""
        if isinstance(data, pd.Series):
            arr = data.values.astype(np.float64)
        else:
            arr = np.asarray(data, dtype=np.float64)
        
        if use_gpu and HAS_CUPY:
            self._ensure_device()  # Vérifier device avant transfert
            return cp.asarray(arr)
        return arr
    
    def _to_numpy(self, arr: Any) -> np.ndarray:
        """Convertit un array en numpy."""
        if HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)
    
    def sma(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Simple Moving Average avec accélération GPU.
        
        Args:
            prices: Array de prix
            period: Période de la moyenne
            
        Returns:
            Array SMA (même taille que prices)
        """
        n = len(prices)
        use_gpu = self._should_use_gpu(n)
        xp = cp if use_gpu and HAS_CUPY else np
        
        arr = self._to_array(prices, use_gpu)
        result = xp.full(n, xp.nan, dtype=xp.float64)
        
        # Calcul rolling mean
        cumsum = xp.cumsum(arr)
        result[period-1:] = (cumsum[period-1:] - xp.concatenate([[0], cumsum[:-period]])) / period
        
        return self._to_numpy(result)
    
    def ema(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Exponential Moving Average avec accélération GPU.
        
        Args:
            prices: Array de prix
            period: Période de l'EMA
            
        Returns:
            Array EMA
        """
        n = len(prices)
        use_gpu = self._should_use_gpu(n)
        xp = cp if use_gpu and HAS_CUPY else np
        
        arr = self._to_array(prices, use_gpu)
        
        # Alpha = 2 / (period + 1)
        alpha = 2.0 / (period + 1)
        
        # EMA récursif (difficile à paralléliser entièrement)
        # On utilise une approximation ou on fait sur CPU si trop petit
        if not use_gpu or n < 10000:
            # CPU version (plus précise pour EMA récursif)
            arr_cpu = self._to_numpy(arr)
            ema = np.zeros(n, dtype=np.float64)
            ema[0] = arr_cpu[0]
            for i in range(1, n):
                ema[i] = alpha * arr_cpu[i] + (1 - alpha) * ema[i-1]
            return ema
        
        # GPU approximation avec filter
        result = xp.zeros(n, dtype=xp.float64)
        result[0] = arr[0]
        
        # Vectorized approximation (moins précis mais plus rapide)
        weights = xp.power(1 - alpha, xp.arange(n))
        for i in range(1, n):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        
        return self._to_numpy(result)
    
    def rsi(self, prices: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """
        Relative Strength Index avec accélération GPU.
        
        Args:
            prices: Array de prix
            period: Période RSI (défaut: 14)
            
        Returns:
            Array RSI (0-100)
        """
        n = len(prices)
        use_gpu = self._should_use_gpu(n)
        xp = cp if use_gpu and HAS_CUPY else np
        
        arr = self._to_array(prices, use_gpu)
        
        # Calcul des deltas
        delta = xp.diff(arr)
        
        # Gains et pertes
        gains = xp.where(delta > 0, delta, 0)
        losses = xp.where(delta < 0, -delta, 0)
        
        # Moyennes mobiles exponentielles
        alpha = 1.0 / period
        
        avg_gain = xp.zeros(n, dtype=xp.float64)
        avg_loss = xp.zeros(n, dtype=xp.float64)
        
        # Premier calcul: moyenne simple
        if n > period:
            avg_gain[period] = xp.mean(gains[:period])
            avg_loss[period] = xp.mean(losses[:period])
            
            # EMA récursif
            for i in range(period + 1, n):
                avg_gain[i] = alpha * gains[i-1] + (1 - alpha) * avg_gain[i-1]
                avg_loss[i] = alpha * losses[i-1] + (1 - alpha) * avg_loss[i-1]
        
        # RSI (éviter division par zéro avec np.errstate)
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = xp.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        
        # NaN pour les premières valeurs
        rsi[:period] = xp.nan
        
        return self._to_numpy(rsi)
    
    def bollinger_bands(
        self,
        prices: Union[np.ndarray, pd.Series],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands avec accélération GPU.
        
        Args:
            prices: Array de prix
            period: Période de la moyenne (défaut: 20)
            std_dev: Nombre d'écarts-types (défaut: 2.0)
            
        Returns:
            Tuple (upper_band, middle_band, lower_band)
        """
        n = len(prices)
        use_gpu = self._should_use_gpu(n)
        xp = cp if use_gpu and HAS_CUPY else np
        
        arr = self._to_array(prices, use_gpu)
        
        # Middle band = SMA
        middle = xp.full(n, xp.nan, dtype=xp.float64)
        std = xp.full(n, xp.nan, dtype=xp.float64)
        
        # Rolling mean et std
        for i in range(period - 1, n):
            window = arr[i - period + 1:i + 1]
            middle[i] = xp.mean(window)
            std[i] = xp.std(window)
        
        # Upper et lower bands
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return (
            self._to_numpy(upper),
            self._to_numpy(middle),
            self._to_numpy(lower)
        )
    
    def atr(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        period: int = 14
    ) -> np.ndarray:
        """
        Average True Range avec accélération GPU.
        
        Args:
            high: Array des plus hauts
            low: Array des plus bas
            close: Array des clôtures
            period: Période ATR (défaut: 14)
            
        Returns:
            Array ATR
        """
        n = len(close)
        use_gpu = self._should_use_gpu(n)
        xp = cp if use_gpu and HAS_CUPY else np
        
        h = self._to_array(high, use_gpu)
        l = self._to_array(low, use_gpu)
        c = self._to_array(close, use_gpu)
        
        # True Range
        tr1 = h - l
        tr2 = xp.abs(h - xp.concatenate([[c[0]], c[:-1]]))
        tr3 = xp.abs(l - xp.concatenate([[c[0]], c[:-1]]))
        
        tr = xp.maximum(tr1, xp.maximum(tr2, tr3))
        
        # ATR = EMA du True Range
        atr = xp.full(n, xp.nan, dtype=xp.float64)
        atr[period-1] = xp.mean(tr[:period])
        
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return self._to_numpy(atr)
    
    def macd(
        self,
        prices: Union[np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD avec accélération GPU.
        
        Args:
            prices: Array de prix
            fast_period: Période EMA rapide (défaut: 12)
            slow_period: Période EMA lente (défaut: 26)
            signal_period: Période ligne signal (défaut: 9)
            
        Returns:
            Tuple (macd_line, signal_line, histogram)
        """
        # EMA rapide et lente
        fast_ema = self.ema(prices, fast_period)
        slow_ema = self.ema(prices, slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA du MACD)
        signal_line = self.ema(macd_line, signal_period)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


# ======================== Benchmark GPU vs CPU ========================

def benchmark_gpu_cpu(n_samples: int = 100000, n_runs: int = 5) -> dict:
    """
    Compare les performances GPU vs CPU.
    
    Args:
        n_samples: Nombre d'échantillons
        n_runs: Nombre de runs pour moyenne
        
    Returns:
        Dict avec timings et speedup
    """
    prices = np.random.randn(n_samples).cumsum() + 100
    
    results = {
        "n_samples": n_samples,
        "n_runs": n_runs,
        "gpu_available": gpu_available(),
    }
    
    # CPU benchmark
    calc_cpu = GPUIndicatorCalculator(use_gpu=False)
    
    cpu_times = []
    for _ in range(n_runs):
        start = time.time()
        calc_cpu.sma(prices, 20)
        calc_cpu.ema(prices, 12)
        calc_cpu.rsi(prices, 14)
        calc_cpu.bollinger_bands(prices, 20, 2.0)
        cpu_times.append(time.time() - start)
    
    results["cpu_avg_time"] = np.mean(cpu_times)
    
    # GPU benchmark (si disponible)
    if gpu_available():
        calc_gpu = GPUIndicatorCalculator(use_gpu=True, min_samples=0)
        
        # Warmup
        calc_gpu.sma(prices[:1000], 20)
        
        gpu_times = []
        for _ in range(n_runs):
            start = time.time()
            calc_gpu.sma(prices, 20)
            calc_gpu.ema(prices, 12)
            calc_gpu.rsi(prices, 14)
            calc_gpu.bollinger_bands(prices, 20, 2.0)
            gpu_times.append(time.time() - start)
        
        results["gpu_avg_time"] = np.mean(gpu_times)
        results["speedup"] = results["cpu_avg_time"] / results["gpu_avg_time"]
    
    return results
