"""
Tests pour Circuit Breaker.
"""

import pytest
import time
import threading
from unittest.mock import Mock


class TestCircuitState:
    """Tests pour les états du circuit."""
    
    def test_states_exist(self):
        """Test que tous les états existent."""
        from utils.circuit_breaker import CircuitState
        
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN


class TestCircuitStats:
    """Tests pour CircuitStats."""
    
    def test_initial_stats(self):
        """Test stats initiales."""
        from utils.circuit_breaker import CircuitStats
        
        stats = CircuitStats()
        
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.success_rate == 1.0
        assert stats.failure_rate == 0.0
    
    def test_record_success(self):
        """Test enregistrement succès."""
        from utils.circuit_breaker import CircuitStats
        
        stats = CircuitStats()
        stats.record_success()
        
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.consecutive_failures == 0
        assert stats.last_success_time is not None
    
    def test_record_failure(self):
        """Test enregistrement échec."""
        from utils.circuit_breaker import CircuitStats
        
        stats = CircuitStats()
        stats.record_failure()
        stats.record_failure()
        
        assert stats.total_calls == 2
        assert stats.failed_calls == 2
        assert stats.consecutive_failures == 2
    
    def test_consecutive_failures_reset_on_success(self):
        """Test que les échecs consécutifs sont réinitialisés."""
        from utils.circuit_breaker import CircuitStats
        
        stats = CircuitStats()
        stats.record_failure()
        stats.record_failure()
        assert stats.consecutive_failures == 2
        
        stats.record_success()
        assert stats.consecutive_failures == 0
    
    def test_to_dict(self):
        """Test export to_dict."""
        from utils.circuit_breaker import CircuitStats
        
        stats = CircuitStats()
        stats.record_success()
        stats.record_failure()
        
        d = stats.to_dict()
        
        assert "total_calls" in d
        assert "success_rate" in d
        assert d["total_calls"] == 2


class TestCircuitBreaker:
    """Tests pour CircuitBreaker."""
    
    def test_initial_state_closed(self):
        """Test état initial = CLOSED."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker("test")
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
    
    def test_successful_call(self):
        """Test appel réussi."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test")
        
        result = breaker.call(lambda: 42)
        
        assert result == 42
        assert breaker.stats.successful_calls == 1
    
    def test_failed_call(self):
        """Test appel échoué."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=5)
        
        def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        assert breaker.stats.failed_calls == 1
        assert breaker.is_closed  # Pas encore ouvert
    
    def test_opens_after_threshold(self):
        """Test ouverture après seuil d'échecs."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        def failing_func():
            raise ValueError("fail")
        
        # 3 échecs = ouverture
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open
    
    def test_rejects_when_open(self):
        """Test rejet des appels quand ouvert."""
        from utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
        
        breaker = CircuitBreaker("test", failure_threshold=1)
        
        # Ouvrir le circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        assert breaker.is_open
        
        # Les appels suivants sont rejetés
        with pytest.raises(CircuitBreakerError):
            breaker.call(lambda: 42)
        
        assert breaker.stats.rejected_calls == 1
    
    def test_half_open_after_timeout(self):
        """Test passage en HALF_OPEN après timeout."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker(
            "test", 
            failure_threshold=1, 
            recovery_timeout=0.1  # 100ms
        )
        
        # Ouvrir
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        assert breaker.is_open
        
        # Attendre le timeout
        time.sleep(0.15)
        
        # Devrait être en HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_closes_on_success_in_half_open(self):
        """Test fermeture après succès en HALF_OPEN."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker(
            "test", 
            failure_threshold=1, 
            recovery_timeout=0.1
        )
        
        # Ouvrir
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        # Attendre transition vers HALF_OPEN
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Appel réussi = fermeture
        result = breaker.call(lambda: "ok")
        
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test réouverture après échec en HALF_OPEN."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker(
            "test", 
            failure_threshold=1, 
            recovery_timeout=0.1
        )
        
        # Ouvrir
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        # Attendre transition vers HALF_OPEN
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Échec en HALF_OPEN = retour OPEN
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail again")))
        
        assert breaker.state == CircuitState.OPEN
    
    def test_decorator_usage(self):
        """Test utilisation comme décorateur."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test")
        
        @breaker
        def my_func(x):
            return x * 2
        
        result = my_func(21)
        
        assert result == 42
        assert breaker.stats.successful_calls == 1
    
    def test_context_manager_success(self):
        """Test context manager - succès."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test")
        
        with breaker:
            result = 42
        
        assert breaker.stats.successful_calls == 1
    
    def test_context_manager_failure(self):
        """Test context manager - échec."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test")
        
        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("fail")
        
        assert breaker.stats.failed_calls == 1
    
    def test_reset(self):
        """Test reset du circuit."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker("test", failure_threshold=1)
        
        # Ouvrir
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        assert breaker.is_open
        
        # Reset
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0
    
    def test_force_open(self):
        """Test force_open pour maintenance."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker("test")
        
        breaker.force_open()
        
        assert breaker.state == CircuitState.OPEN
    
    def test_excluded_exceptions(self):
        """Test exceptions exclues."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker(
            "test", 
            failure_threshold=1,
            excluded_exceptions=[KeyError]
        )
        
        # KeyError est exclu - ne compte pas comme échec
        with pytest.raises(KeyError):
            breaker.call(lambda: (_ for _ in ()).throw(KeyError("excluded")))
        
        assert breaker.stats.failed_calls == 0
        assert breaker.is_closed
    
    def test_thread_safety(self):
        """Test thread safety."""
        from utils.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=100)
        results = []
        
        def worker():
            for _ in range(50):
                try:
                    breaker.call(lambda: 1)
                    results.append(1)
                except Exception:
                    pass
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Tous les appels devraient réussir
        assert len(results) == 200
        assert breaker.stats.successful_calls == 200


class TestCircuitBreakerRegistry:
    """Tests pour CircuitBreakerRegistry."""
    
    def test_get_or_create(self):
        """Test get_or_create."""
        from utils.circuit_breaker import CircuitBreakerRegistry
        
        registry = CircuitBreakerRegistry()
        registry.clear()  # Nettoyer pour le test
        
        breaker1 = registry.get_or_create("test_registry")
        breaker2 = registry.get_or_create("test_registry")
        
        assert breaker1 is breaker2
    
    def test_list_all(self):
        """Test list_all."""
        from utils.circuit_breaker import CircuitBreakerRegistry
        
        registry = CircuitBreakerRegistry()
        registry.clear()
        
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")
        
        names = registry.list_all()
        
        assert "breaker1" in names
        assert "breaker2" in names
    
    def test_get_all_stats(self):
        """Test get_all_stats."""
        from utils.circuit_breaker import CircuitBreakerRegistry
        
        registry = CircuitBreakerRegistry()
        registry.clear()
        
        breaker = registry.get_or_create("test_stats")
        breaker.call(lambda: 1)
        
        stats = registry.get_all_stats()
        
        assert "test_stats" in stats
        assert stats["test_stats"]["successful_calls"] == 1


class TestCircuitBreakerShortcuts:
    """Tests pour les fonctions raccourcis."""
    
    def test_get_circuit_breaker(self):
        """Test get_circuit_breaker."""
        from utils.circuit_breaker import get_circuit_breaker
        
        breaker = get_circuit_breaker("shortcut_test", failure_threshold=10)
        
        assert breaker.name == "shortcut_test"
        assert breaker.failure_threshold == 10
    
    def test_circuit_breaker_decorator(self):
        """Test circuit_breaker comme décorateur."""
        from utils.circuit_breaker import circuit_breaker
        
        @circuit_breaker("decorator_test", failure_threshold=5)
        def my_protected_func(x):
            return x + 1
        
        result = my_protected_func(10)
        
        assert result == 11
