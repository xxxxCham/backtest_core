"""
Tests pour les modules Phase 2 et Phase 4 restants:
- Device Agnostic Backend
- Error Recovery
- GPU OOM Handler
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# ============================================================
# Tests Device Backend
# ============================================================

from performance.device_backend import (
    DeviceType,
    DeviceInfo,
    ArrayBackend,
    get_backend,
    use_gpu,
    use_cpu,
    is_gpu_available,
    get_device_info,
    gpu_context,
    cpu_context,
    array_like,
)


class TestDeviceType:
    """Tests pour DeviceType."""
    
    def test_device_types_exist(self):
        """Test existence des types."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.GPU.value == "gpu"
        assert DeviceType.AUTO.value == "auto"


class TestDeviceInfo:
    """Tests pour DeviceInfo."""
    
    def test_cpu_info(self):
        """Test info CPU."""
        info = DeviceInfo(
            device_type=DeviceType.CPU,
            name="Test CPU",
        )
        
        assert "CPU" in str(info)
        assert info.device_type == DeviceType.CPU
    
    def test_gpu_info_with_memory(self):
        """Test info GPU avec mémoire."""
        info = DeviceInfo(
            device_type=DeviceType.GPU,
            name="Test GPU",
            memory_total=8 * 1024**3,  # 8 GB
            memory_free=4 * 1024**3,
        )
        
        assert "GPU" in str(info)
        assert "8.0 GB" in str(info)


class TestArrayBackend:
    """Tests pour ArrayBackend."""
    
    def test_singleton(self):
        """Test pattern singleton."""
        backend1 = ArrayBackend()
        backend2 = ArrayBackend()
        
        assert backend1 is backend2
    
    def test_xp_returns_numpy(self):
        """Test que xp retourne numpy par défaut."""
        backend = get_backend()
        
        # Si pas de GPU, doit être numpy
        if not backend.gpu_available:
            assert backend.xp is np
    
    def test_array_creation(self):
        """Test création d'arrays."""
        backend = get_backend()
        
        arr = backend.array([1, 2, 3])
        assert len(arr) == 3
        
        zeros = backend.zeros((3, 4))
        assert zeros.shape == (3, 4)
        
        ones = backend.ones((2, 2))
        assert ones.sum() == 4
    
    def test_math_operations(self):
        """Test opérations mathématiques."""
        backend = get_backend()
        
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        
        assert backend.sum(arr) == 10.0
        assert backend.mean(arr) == 2.5
        assert backend.min(arr) == 1.0
        assert backend.max(arr) == 4.0
    
    def test_rolling_mean(self):
        """Test moyenne mobile."""
        backend = get_backend()
        
        data = backend.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = backend.rolling_mean(data, window=3)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
    
    def test_to_numpy(self):
        """Test conversion vers numpy."""
        backend = get_backend()
        
        arr = backend.array([1, 2, 3])
        numpy_arr = backend.to_numpy(arr)
        
        assert isinstance(numpy_arr, np.ndarray)
    
    def test_memory_info(self):
        """Test infos mémoire."""
        backend = get_backend()
        
        info = backend.memory_info()
        
        assert "device" in info
        assert "free" in info
        assert "total" in info
    
    def test_use_device_cpu(self):
        """Test changement vers CPU."""
        backend = get_backend()
        
        success = backend.use_device(DeviceType.CPU)
        assert success is True
        assert backend.device_type == DeviceType.CPU


class TestBackendHelpers:
    """Tests pour les fonctions helper."""
    
    def test_get_backend(self):
        """Test get_backend retourne singleton."""
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2
    
    def test_use_cpu(self):
        """Test forçage CPU."""
        result = use_cpu()
        assert result is True
        assert get_backend().device_type == DeviceType.CPU
    
    def test_get_device_info(self):
        """Test récupération info device."""
        info = get_device_info()
        assert isinstance(info, DeviceInfo)
    
    def test_array_like(self):
        """Test création array via helper."""
        arr = array_like([1, 2, 3])
        assert len(arr) == 3


# ============================================================
# Tests Error Recovery
# ============================================================

from utils.error_recovery import (
    ErrorCategory,
    ErrorInfo,
    RetryConfig,
    ErrorClassifier,
    RetryHandler,
    RecoveryStrategy,
    with_retry,
    retry_on_memory_error,
)


class TestErrorCategory:
    """Tests pour ErrorCategory."""
    
    def test_categories_exist(self):
        """Test existence des catégories."""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.NETWORK.value == "network"


class TestErrorInfo:
    """Tests pour ErrorInfo."""
    
    def test_creation(self):
        """Test création info erreur."""
        exc = ValueError("test error")
        info = ErrorInfo(
            exception=exc,
            category=ErrorCategory.VALIDATION,
            timestamp=time.time(),
            attempt=1,
        )
        
        assert info.exception is exc
        assert info.category == ErrorCategory.VALIDATION
        assert info.attempt == 1


class TestErrorClassifier:
    """Tests pour ErrorClassifier."""
    
    def test_classify_connection_error(self):
        """Test classification erreur connexion."""
        classifier = ErrorClassifier()
        exc = ConnectionError("connection refused")
        
        category = classifier.classify(exc)
        assert category == ErrorCategory.NETWORK
    
    def test_classify_memory_error(self):
        """Test classification erreur mémoire."""
        classifier = ErrorClassifier()
        exc = MemoryError("out of memory")
        
        category = classifier.classify(exc)
        assert category == ErrorCategory.RESOURCE
    
    def test_classify_value_error(self):
        """Test classification erreur valeur."""
        classifier = ErrorClassifier()
        exc = ValueError("invalid value")
        
        category = classifier.classify(exc)
        assert category == ErrorCategory.VALIDATION
    
    def test_classify_by_message(self):
        """Test classification par message."""
        classifier = ErrorClassifier()
        exc = Exception("network timeout occurred")
        
        category = classifier.classify(exc)
        assert category == ErrorCategory.NETWORK
    
    def test_is_retryable(self):
        """Test détermination retryable."""
        classifier = ErrorClassifier()
        
        assert classifier.is_retryable(ConnectionError()) is True
        assert classifier.is_retryable(MemoryError()) is True
        assert classifier.is_retryable(ValueError()) is False


class TestRetryConfig:
    """Tests pour RetryConfig."""
    
    def test_defaults(self):
        """Test valeurs par défaut."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.exponential_base == 2.0


class TestRetryHandler:
    """Tests pour RetryHandler."""
    
    def test_successful_execution(self):
        """Test exécution réussie sans retry."""
        handler = RetryHandler()
        
        def success():
            return "ok"
        
        result = handler.execute(success)
        assert result == "ok"
    
    def test_retry_on_transient_error(self):
        """Test retry sur erreur transitoire."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        handler = RetryHandler(config=config)
        
        call_count = [0]
        
        def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return "success"
        
        result = handler.execute(fails_twice)
        
        assert result == "success"
        assert call_count[0] == 3
    
    def test_no_retry_on_permanent_error(self):
        """Test pas de retry sur erreur permanente."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        handler = RetryHandler(config=config)
        
        call_count = [0]
        
        def permanent_fail():
            call_count[0] += 1
            raise ValueError("permanent")
        
        with pytest.raises(ValueError):
            handler.execute(permanent_fail)
        
        assert call_count[0] == 1  # Pas de retry
    
    def test_retry_decorator(self):
        """Test décorateur retry."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        handler = RetryHandler(config=config)
        
        call_count = [0]
        
        @handler.retry
        def unstable():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError()
            return "done"
        
        result = unstable()
        
        assert result == "done"
        assert call_count[0] == 2
    
    def test_on_retry_callback(self):
        """Test callback on_retry."""
        callbacks = []
        
        def on_retry(error_info):
            callbacks.append(error_info)
        
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        handler = RetryHandler(config=config, on_retry=on_retry)
        
        call_count = [0]
        
        def fails_once():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError()
            return "ok"
        
        handler.execute(fails_once)
        
        assert len(callbacks) == 1
    
    def test_get_errors(self):
        """Test récupération historique erreurs."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        handler = RetryHandler(config=config)
        
        def fails():
            raise ConnectionError("test")
        
        try:
            handler.execute(fails)
        except ConnectionError:
            pass
        
        errors = handler.get_errors()
        assert len(errors) == 2


class TestRecoveryStrategy:
    """Tests pour RecoveryStrategy."""
    
    def test_recover_resource_error(self):
        """Test récupération erreur ressource."""
        strategy = RecoveryStrategy()
        
        error_info = ErrorInfo(
            exception=MemoryError("oom"),
            category=ErrorCategory.RESOURCE,
            timestamp=time.time(),
        )
        
        recovered = strategy.recover(error_info)
        assert recovered is True
    
    def test_no_recovery_permanent(self):
        """Test pas de récupération erreur permanente."""
        strategy = RecoveryStrategy()
        
        error_info = ErrorInfo(
            exception=ValueError("bad"),
            category=ErrorCategory.PERMANENT,
            timestamp=time.time(),
        )
        
        recovered = strategy.recover(error_info)
        assert recovered is False
    
    def test_custom_strategy(self):
        """Test stratégie personnalisée."""
        strategy = RecoveryStrategy()
        
        custom_called = [False]
        
        def custom_handler(error):
            custom_called[0] = True
            return True
        
        strategy.register(ErrorCategory.VALIDATION, custom_handler)
        
        error_info = ErrorInfo(
            exception=ValueError("test"),
            category=ErrorCategory.VALIDATION,
            timestamp=time.time(),
        )
        
        strategy.recover(error_info)
        assert custom_called[0] is True


class TestWithRetryDecorator:
    """Tests pour le décorateur with_retry."""
    
    def test_basic_usage(self):
        """Test utilisation basique."""
        call_count = [0]
        
        @with_retry(max_attempts=3, delay=0.01)
        def unstable():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError()
            return "done"
        
        result = unstable()
        
        assert result == "done"
        assert call_count[0] == 2


# ============================================================
# Tests GPU OOM Handler
# ============================================================

from utils.gpu_oom import (
    MemoryStatus,
    GPUMemoryInfo,
    GPUOOMHandler,
    get_oom_handler,
    safe_gpu,
    gpu_memory_guard,
    clear_gpu_memory,
    get_gpu_memory_status,
)


class TestMemoryStatus:
    """Tests pour MemoryStatus."""
    
    def test_status_values(self):
        """Test valeurs de status."""
        assert MemoryStatus.OK.value == "ok"
        assert MemoryStatus.LOW.value == "low"
        assert MemoryStatus.CRITICAL.value == "critical"
        assert MemoryStatus.OOM.value == "oom"


class TestGPUMemoryInfo:
    """Tests pour GPUMemoryInfo."""
    
    def test_defaults(self):
        """Test valeurs par défaut."""
        info = GPUMemoryInfo()
        
        assert info.total == 0
        assert info.status == MemoryStatus.UNAVAILABLE
    
    def test_usage_percent(self):
        """Test calcul pourcentage."""
        info = GPUMemoryInfo(
            total=1000,
            used=250,
            free=750,
            status=MemoryStatus.OK,
        )
        
        assert info.usage_percent == 25.0
    
    def test_free_mb(self):
        """Test conversion MB."""
        info = GPUMemoryInfo(
            total=1024 * 1024 * 1024,  # 1 GB
            used=0,
            free=1024 * 1024 * 1024,
            status=MemoryStatus.OK,
        )
        
        assert info.free_mb == 1024.0
        assert info.free_gb == 1.0


class TestGPUOOMHandler:
    """Tests pour GPUOOMHandler."""
    
    def test_creation(self):
        """Test création handler."""
        handler = GPUOOMHandler()
        
        assert handler.low_threshold == 0.2
        assert handler.critical_threshold == 0.1
    
    def test_get_memory_info_without_gpu(self):
        """Test info mémoire sans GPU."""
        handler = GPUOOMHandler()
        
        info = handler.get_memory_info()
        
        # Sans CuPy/GPU installé
        if not handler._gpu_available:
            assert info.status == MemoryStatus.UNAVAILABLE
    
    def test_estimate_memory_required(self):
        """Test estimation mémoire."""
        handler = GPUOOMHandler()
        
        # Array 1000x1000 float64
        estimate = handler.estimate_memory_required((1000, 1000), np.float64)
        
        # 1M * 8 bytes * 1.2 overhead = 9.6 MB
        expected = 1000 * 1000 * 8 * 1.2
        assert estimate == int(expected)
    
    def test_clear_memory_without_gpu(self):
        """Test nettoyage sans GPU."""
        handler = GPUOOMHandler()
        
        if not handler._gpu_available:
            result = handler.clear_memory()
            assert result is False
    
    def test_safe_gpu_operation_decorator(self):
        """Test décorateur safe_gpu_operation."""
        handler = GPUOOMHandler(fallback_to_cpu=True)
        
        @handler.safe_gpu_operation
        def compute(x):
            return np.sum(x)
        
        result = compute(np.array([1, 2, 3]))
        assert result == 6
    
    def test_get_stats(self):
        """Test statistiques."""
        handler = GPUOOMHandler()
        
        stats = handler.get_stats()
        
        assert "gpu_available" in stats
        assert "oom_count" in stats
        assert "fallback_count" in stats
        assert "memory_status" in stats


class TestOOMHelpers:
    """Tests pour les fonctions helper OOM."""
    
    def test_get_oom_handler_singleton(self):
        """Test singleton OOM handler."""
        h1 = get_oom_handler()
        h2 = get_oom_handler()
        
        assert h1 is h2
    
    def test_safe_gpu_decorator(self):
        """Test décorateur safe_gpu."""
        @safe_gpu
        def my_func(x):
            return x * 2
        
        result = my_func(5)
        assert result == 10
    
    def test_get_gpu_memory_status(self):
        """Test récupération status mémoire."""
        info = get_gpu_memory_status()
        
        assert isinstance(info, GPUMemoryInfo)


class TestGPUMemoryGuard:
    """Tests pour gpu_memory_guard."""
    
    def test_guard_success(self):
        """Test guard réussi."""
        handler = GPUOOMHandler()
        
        # Si pas de GPU, le guard devrait juste logger un warning
        if not handler._gpu_available:
            with handler.memory_guard(required_mb=100):
                result = 1 + 1
            assert result == 2
