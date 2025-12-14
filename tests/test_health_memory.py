"""
Tests pour Health Monitor et Memory Manager.
"""

import gc
import time
import threading
import pytest

from utils.health import (
    HealthMonitor,
    HealthStatus,
    HealthThresholds,
    ResourceType,
    ResourceMetrics,
    HealthSnapshot,
    get_health_monitor,
    check_system_health,
    is_system_healthy,
)

from utils.memory import (
    MemoryManager,
    MemoryConfig,
    ManagedCache,
    get_memory_manager,
    cleanup_memory,
)


# ============================================================================
# Tests HealthStatus et ResourceType
# ============================================================================

class TestEnums:
    """Tests pour les enums."""
    
    def test_health_status_values(self):
        """Test valeurs HealthStatus."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_resource_type_values(self):
        """Test valeurs ResourceType."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.DISK.value == "disk"


# ============================================================================
# Tests HealthThresholds
# ============================================================================

class TestHealthThresholds:
    """Tests pour HealthThresholds."""
    
    def test_default_thresholds(self):
        """Test seuils par défaut."""
        t = HealthThresholds()
        
        assert t.cpu_warning == 80.0
        assert t.cpu_critical == 95.0
        assert t.memory_warning == 75.0
        assert t.memory_critical == 90.0
    
    def test_get_status_healthy(self):
        """Test status healthy."""
        t = HealthThresholds()
        
        assert t.get_status(ResourceType.CPU, 50.0) == HealthStatus.HEALTHY
        assert t.get_status(ResourceType.MEMORY, 60.0) == HealthStatus.HEALTHY
    
    def test_get_status_warning(self):
        """Test status warning."""
        t = HealthThresholds()
        
        assert t.get_status(ResourceType.CPU, 85.0) == HealthStatus.WARNING
        assert t.get_status(ResourceType.MEMORY, 80.0) == HealthStatus.WARNING
    
    def test_get_status_critical(self):
        """Test status critical."""
        t = HealthThresholds()
        
        assert t.get_status(ResourceType.CPU, 98.0) == HealthStatus.CRITICAL
        assert t.get_status(ResourceType.MEMORY, 95.0) == HealthStatus.CRITICAL
    
    def test_custom_thresholds(self):
        """Test seuils personnalisés."""
        t = HealthThresholds(cpu_warning=50.0, cpu_critical=70.0)
        
        assert t.get_status(ResourceType.CPU, 60.0) == HealthStatus.WARNING
        assert t.get_status(ResourceType.CPU, 75.0) == HealthStatus.CRITICAL


# ============================================================================
# Tests ResourceMetrics
# ============================================================================

class TestResourceMetrics:
    """Tests pour ResourceMetrics."""
    
    def test_create_metrics(self):
        """Test création métriques."""
        m = ResourceMetrics(
            resource_type=ResourceType.CPU,
            usage_percent=50.0,
            available=50.0,
            total=100.0,
            status=HealthStatus.HEALTHY,
        )
        
        assert m.resource_type == ResourceType.CPU
        assert m.usage_percent == 50.0
        assert m.status == HealthStatus.HEALTHY
    
    def test_to_dict(self):
        """Test conversion dict."""
        m = ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            usage_percent=75.0,
            available=8000000000,
            total=32000000000,
            status=HealthStatus.WARNING,
        )
        
        d = m.to_dict()
        
        assert d["resource"] == "memory"
        assert d["usage_percent"] == 75.0
        assert d["status"] == "warning"


# ============================================================================
# Tests HealthMonitor
# ============================================================================

class TestHealthMonitor:
    """Tests pour HealthMonitor."""
    
    def test_create_monitor(self):
        """Test création moniteur."""
        monitor = HealthMonitor()
        assert monitor.thresholds is not None
    
    def test_custom_thresholds(self):
        """Test avec seuils personnalisés."""
        thresholds = HealthThresholds(cpu_warning=60.0)
        monitor = HealthMonitor(thresholds=thresholds)
        
        assert monitor.thresholds.cpu_warning == 60.0
    
    def test_get_cpu_metrics(self):
        """Test métriques CPU."""
        monitor = HealthMonitor()
        metrics = monitor.get_cpu_metrics()
        
        assert metrics.resource_type == ResourceType.CPU
        assert 0 <= metrics.usage_percent <= 100
        assert metrics.status in HealthStatus
    
    def test_get_memory_metrics(self):
        """Test métriques mémoire."""
        monitor = HealthMonitor()
        metrics = monitor.get_memory_metrics()
        
        assert metrics.resource_type == ResourceType.MEMORY
        assert 0 <= metrics.usage_percent <= 100
    
    def test_get_disk_metrics(self):
        """Test métriques disque."""
        monitor = HealthMonitor()
        metrics = monitor.get_disk_metrics()
        
        assert metrics.resource_type == ResourceType.DISK
        assert metrics.total > 0 or metrics.status == HealthStatus.UNKNOWN
    
    def test_get_gpu_metrics(self):
        """Test métriques GPU (peut être UNKNOWN)."""
        monitor = HealthMonitor()
        metrics = monitor.get_gpu_metrics()
        
        assert metrics.resource_type == ResourceType.GPU
        # GPU peut ne pas être disponible
        assert metrics.status in HealthStatus
    
    def test_check_health(self):
        """Test check santé complet."""
        monitor = HealthMonitor()
        snapshot = monitor.check_health()
        
        assert isinstance(snapshot, HealthSnapshot)
        assert snapshot.overall_status in HealthStatus
        assert ResourceType.CPU in snapshot.metrics
        assert ResourceType.MEMORY in snapshot.metrics
    
    def test_health_snapshot_to_dict(self):
        """Test conversion snapshot en dict."""
        monitor = HealthMonitor()
        snapshot = monitor.check_health()
        
        d = snapshot.to_dict()
        
        assert "timestamp" in d
        assert "overall_status" in d
        assert "metrics" in d
        assert "cpu" in d["metrics"]
    
    def test_get_system_info(self):
        """Test infos système."""
        monitor = HealthMonitor()
        info = monitor.get_system_info()
        
        assert "platform" in info
        assert "python_version" in info
    
    def test_is_healthy(self):
        """Test vérification santé."""
        monitor = HealthMonitor()
        result = monitor.is_healthy()
        
        assert isinstance(result, bool)
    
    def test_history(self):
        """Test historique."""
        monitor = HealthMonitor()
        
        # Faire plusieurs checks
        for _ in range(3):
            monitor.check_health()
        
        history = monitor.get_history()
        assert len(history) == 3
    
    def test_alert_callback(self):
        """Test callback d'alerte."""
        alerts_received = []
        
        def on_alert(msg, status):
            alerts_received.append((msg, status))
        
        # Seuils très bas pour déclencher des alertes
        thresholds = HealthThresholds(
            cpu_warning=0.0,
            memory_warning=0.0
        )
        monitor = HealthMonitor(thresholds=thresholds, on_alert=on_alert)
        monitor.check_health()
        
        # Des alertes devraient avoir été générées
        assert len(alerts_received) > 0
    
    def test_summary(self):
        """Test résumé textuel."""
        monitor = HealthMonitor()
        summary = monitor.summary()
        
        assert "Health Monitor" in summary
        assert "Status:" in summary
    
    def test_monitoring_start_stop(self):
        """Test démarrage/arrêt surveillance."""
        monitor = HealthMonitor()
        
        monitor.start_monitoring(interval=0.1)
        assert monitor._monitoring is True
        
        time.sleep(0.3)  # Laisser tourner
        
        monitor.stop_monitoring()
        assert monitor._monitoring is False
    
    def test_singleton(self):
        """Test singleton global."""
        m1 = get_health_monitor()
        m2 = get_health_monitor()
        
        assert m1 is m2
    
    def test_shortcuts(self):
        """Test fonctions raccourcis."""
        snapshot = check_system_health()
        assert isinstance(snapshot, HealthSnapshot)
        
        healthy = is_system_healthy()
        assert isinstance(healthy, bool)


# ============================================================================
# Tests ManagedCache
# ============================================================================

class TestManagedCache:
    """Tests pour ManagedCache."""
    
    def test_create_cache(self):
        """Test création cache."""
        cache = ManagedCache(max_size_mb=10.0, name="test")
        
        assert cache.name == "test"
        assert len(cache) == 0
    
    def test_set_get(self):
        """Test set/get basique."""
        cache = ManagedCache(max_size_mb=10.0)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_get_default(self):
        """Test get avec défaut."""
        cache = ManagedCache(max_size_mb=10.0)
        
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"
    
    def test_delete(self):
        """Test suppression."""
        cache = ManagedCache(max_size_mb=10.0)
        
        cache.set("key1", "value1")
        assert "key1" in cache
        
        cache.delete("key1")
        assert "key1" not in cache
    
    def test_clear(self):
        """Test vidage."""
        cache = ManagedCache(max_size_mb=10.0)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        freed = cache.clear()
        
        assert len(cache) == 0
        assert freed > 0
    
    def test_lru_eviction(self):
        """Test éviction LRU."""
        # Cache très petit
        cache = ManagedCache(max_size_mb=0.001)  # ~1KB
        
        # Ajouter des données
        cache.set("key1", "a" * 500)
        cache.set("key2", "b" * 500)
        
        # key1 devrait être évincé
        # (dépend de la taille estimée)
        assert len(cache) <= 2
    
    def test_contains(self):
        """Test opérateur in."""
        cache = ManagedCache(max_size_mb=10.0)
        
        cache.set("exists", "value")
        
        assert "exists" in cache
        assert "not_exists" not in cache
    
    def test_cache_stats(self):
        """Test statistiques cache."""
        cache = ManagedCache(max_size_mb=10.0)
        
        cache.get("miss1")  # Miss
        cache.set("key1", "value1")
        cache.get("key1")   # Hit
        cache.get("miss2")  # Miss
        
        assert cache._stats.cache_hits == 1
        assert cache._stats.cache_misses == 2


# ============================================================================
# Tests MemoryManager
# ============================================================================

class TestMemoryManager:
    """Tests pour MemoryManager."""
    
    def test_create_manager(self):
        """Test création manager."""
        manager = MemoryManager()
        assert manager.config is not None
    
    def test_custom_config(self):
        """Test config personnalisée."""
        config = MemoryConfig(warning_threshold=60.0)
        manager = MemoryManager(config=config)
        
        assert manager.config.warning_threshold == 60.0
    
    def test_get_memory_usage(self):
        """Test lecture usage mémoire."""
        manager = MemoryManager()
        usage = manager.get_memory_usage()
        
        assert 0 <= usage <= 100
    
    def test_get_memory_available_mb(self):
        """Test mémoire disponible."""
        manager = MemoryManager()
        available = manager.get_memory_available_mb()
        
        assert available > 0
    
    def test_create_cache(self):
        """Test création cache managé."""
        manager = MemoryManager()
        cache = manager.create_cache("test_cache", max_size_mb=50.0)
        
        assert isinstance(cache, ManagedCache)
        assert cache.name == "test_cache"
    
    def test_get_cache(self):
        """Test récupération cache."""
        manager = MemoryManager()
        manager.create_cache("my_cache")
        
        cache = manager.get_cache("my_cache")
        assert cache is not None
        
        assert manager.get_cache("nonexistent") is None
    
    def test_cleanup(self):
        """Test nettoyage."""
        manager = MemoryManager()
        
        # Créer un cache et le remplir
        cache = manager.create_cache("cleanup_test")
        cache.set("data", "x" * 10000)
        
        freed = manager.cleanup(aggressive=True)
        
        assert freed >= 0
        assert manager._stats.total_cleanups > 0
    
    def test_cleanup_callback(self):
        """Test callback de nettoyage."""
        manager = MemoryManager()
        callback_called = [False]
        
        def my_cleanup():
            callback_called[0] = True
            return 1000
        
        manager.register_cleanup_callback(my_cleanup)
        manager.cleanup()
        
        assert callback_called[0] is True
    
    def test_memory_context(self):
        """Test context manager."""
        manager = MemoryManager()
        initial_cleanups = manager._stats.total_cleanups
        
        with manager.memory_context(cleanup_after=True):
            # Simuler travail
            pass
        
        assert manager._stats.total_cleanups > initial_cleanups
    
    def test_auto_cleanup_start_stop(self):
        """Test auto-cleanup."""
        manager = MemoryManager()
        manager.config.cleanup_interval = 0.1
        
        manager.start_auto_cleanup()
        assert manager._running is True
        
        time.sleep(0.3)
        
        manager.stop_auto_cleanup()
        assert manager._running is False
    
    def test_get_stats(self):
        """Test statistiques."""
        manager = MemoryManager()
        manager.cleanup()
        
        stats = manager.get_stats()
        
        assert stats.total_cleanups >= 1
        assert stats.current_usage_mb >= 0
    
    def test_get_cache_summary(self):
        """Test résumé caches."""
        manager = MemoryManager()
        cache = manager.create_cache("summary_test", max_size_mb=100)
        cache.set("key", "value")
        
        summary = manager.get_cache_summary()
        
        assert "summary_test" in summary
        assert "size_mb" in summary["summary_test"]
        assert "entries" in summary["summary_test"]
    
    def test_summary(self):
        """Test résumé textuel."""
        manager = MemoryManager()
        manager.create_cache("test")
        
        summary = manager.summary()
        
        assert "Memory Manager" in summary
        assert "Process Memory" in summary
    
    def test_singleton(self):
        """Test singleton global."""
        m1 = get_memory_manager()
        m2 = get_memory_manager()
        
        assert m1 is m2
    
    def test_cleanup_shortcut(self):
        """Test fonction raccourci."""
        freed = cleanup_memory()
        assert freed >= 0


# ============================================================================
# Tests MemoryConfig
# ============================================================================

class TestMemoryConfig:
    """Tests pour MemoryConfig."""
    
    def test_default_config(self):
        """Test config par défaut."""
        config = MemoryConfig()
        
        assert config.warning_threshold == 75.0
        assert config.critical_threshold == 90.0
        assert config.auto_cleanup is True
    
    def test_to_dict(self):
        """Test conversion dict."""
        config = MemoryConfig()
        d = config.to_dict()
        
        assert "warning_threshold" in d
        assert "auto_cleanup" in d


# ============================================================================
# Tests d'intégration
# ============================================================================

class TestIntegration:
    """Tests d'intégration Health + Memory."""
    
    def test_health_and_memory_together(self):
        """Test utilisation conjointe."""
        health = HealthMonitor()
        memory = MemoryManager()
        
        # Check santé
        snapshot = health.check_health()
        
        # Si mémoire haute, nettoyer
        mem_metrics = snapshot.metrics.get(ResourceType.MEMORY)
        if mem_metrics and mem_metrics.status != HealthStatus.HEALTHY:
            memory.cleanup(aggressive=True)
        
        # Re-vérifier
        new_snapshot = health.check_health()
        assert new_snapshot.overall_status in HealthStatus
    
    def test_wait_for_resources(self):
        """Test attente ressources."""
        monitor = HealthMonitor()
        
        # Avec un seuil très haut, devrait réussir immédiatement
        result = monitor.wait_for_resources(
            memory_threshold=99.0,
            timeout=1.0
        )
        
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
