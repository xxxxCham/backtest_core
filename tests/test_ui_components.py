"""
Tests pour les composants UI Phase 5.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


# ============================================================
# Tests System Monitor
# ============================================================

from ui.components.monitor import (
    ResourceReading,
    SystemMonitorConfig,
    SystemMonitor,
)


class TestResourceReading:
    """Tests pour ResourceReading."""
    
    def test_creation(self):
        """Test création d'une lecture."""
        reading = ResourceReading(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
        )
        
        assert reading.cpu_percent == 50.0
        assert reading.memory_percent == 60.0
    
    def test_default_values(self):
        """Test valeurs par défaut."""
        reading = ResourceReading(timestamp=datetime.now())
        
        assert reading.cpu_percent == 0.0
        assert reading.memory_percent == 0.0
        assert reading.gpu_percent == 0.0


class TestSystemMonitorConfig:
    """Tests pour SystemMonitorConfig."""
    
    def test_default_thresholds(self):
        """Test seuils par défaut."""
        config = SystemMonitorConfig()
        
        assert config.cpu_warning == 80.0
        assert config.cpu_critical == 95.0
        assert config.memory_warning == 80.0
        assert config.memory_critical == 90.0
    
    def test_custom_thresholds(self):
        """Test seuils personnalisés."""
        config = SystemMonitorConfig(
            cpu_warning=70.0,
            cpu_critical=90.0,
        )
        
        assert config.cpu_warning == 70.0
        assert config.cpu_critical == 90.0


class TestSystemMonitor:
    """Tests pour SystemMonitor."""
    
    def test_creation(self):
        """Test création du moniteur."""
        monitor = SystemMonitor()
        
        assert monitor.config is not None
        assert len(monitor.history) == 0
    
    def test_get_current_reading(self):
        """Test lecture actuelle."""
        monitor = SystemMonitor()
        
        reading = monitor.get_current_reading()
        
        assert isinstance(reading, ResourceReading)
        assert reading.timestamp is not None
        # CPU doit être >= 0
        assert reading.cpu_percent >= 0
    
    def test_update_adds_to_history(self):
        """Test que update ajoute à l'historique."""
        monitor = SystemMonitor()
        
        monitor.update()
        monitor.update()
        
        assert len(monitor.history) == 2
    
    def test_history_limit(self):
        """Test limite de l'historique."""
        config = SystemMonitorConfig(max_history=5)
        monitor = SystemMonitor(config=config)
        
        for _ in range(10):
            monitor.update()
        
        assert len(monitor.history) == 5
    
    def test_get_status_ok(self):
        """Test status OK."""
        monitor = SystemMonitor()
        
        reading = ResourceReading(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_percent=50.0,
        )
        
        status = monitor.get_status(reading)
        
        assert status['cpu'] == 'ok'
        assert status['memory'] == 'ok'
        assert status['disk'] == 'ok'
    
    def test_get_status_warning(self):
        """Test status warning."""
        monitor = SystemMonitor()
        
        reading = ResourceReading(
            timestamp=datetime.now(),
            cpu_percent=85.0,
            memory_percent=85.0,
        )
        
        status = monitor.get_status(reading)
        
        assert status['cpu'] == 'warning'
        assert status['memory'] == 'warning'
    
    def test_get_status_critical(self):
        """Test status critical."""
        monitor = SystemMonitor()
        
        reading = ResourceReading(
            timestamp=datetime.now(),
            cpu_percent=98.0,
            memory_percent=95.0,
        )
        
        status = monitor.get_status(reading)
        
        assert status['cpu'] == 'critical'
        assert status['memory'] == 'critical'
    
    def test_clear_history(self):
        """Test effacement de l'historique."""
        monitor = SystemMonitor()
        
        monitor.update()
        monitor.update()
        assert len(monitor.history) == 2
        
        monitor.clear_history()
        assert len(monitor.history) == 0


# ============================================================
# Tests Sweep Monitor
# ============================================================

from ui.components.sweep_monitor import (
    SweepResult,
    SweepStats,
    SweepMonitor,
)


class TestSweepResult:
    """Tests pour SweepResult."""
    
    def test_creation(self):
        """Test création d'un résultat."""
        result = SweepResult(
            params={"fast": 10, "slow": 20},
            metrics={"sharpe_ratio": 1.5, "total_return": 0.25},
        )
        
        assert result.params["fast"] == 10
        assert result.sharpe == 1.5
        assert result.total_return == 0.25
    
    def test_default_metrics(self):
        """Test métriques par défaut."""
        result = SweepResult(
            params={},
            metrics={},
        )
        
        assert result.sharpe == 0.0
        assert result.total_return == 0.0


class TestSweepStats:
    """Tests pour SweepStats."""
    
    def test_progress_percent(self):
        """Test calcul du pourcentage."""
        stats = SweepStats(
            total_combinations=100,
            evaluated=25,
        )
        
        assert stats.progress_percent == 25.0
    
    def test_progress_percent_zero_total(self):
        """Test pourcentage avec total = 0."""
        stats = SweepStats(
            total_combinations=0,
            evaluated=0,
        )
        
        assert stats.progress_percent == 0.0
    
    def test_elapsed(self):
        """Test temps écoulé."""
        stats = SweepStats(
            start_time=datetime.now() - timedelta(seconds=10)
        )
        
        assert stats.elapsed_seconds >= 10
    
    def test_rate(self):
        """Test taux d'évaluation."""
        stats = SweepStats(
            evaluated=100,
            start_time=datetime.now() - timedelta(seconds=10)
        )
        
        assert stats.rate >= 9.0  # ~10 eval/sec
    
    def test_eta_str(self):
        """Test formatage ETA."""
        stats = SweepStats(
            total_combinations=200,
            evaluated=100,
            start_time=datetime.now() - timedelta(seconds=10)
        )
        
        eta = stats.eta_str
        assert isinstance(eta, str)
        assert len(eta) > 0


class TestSweepMonitor:
    """Tests pour SweepMonitor."""
    
    def test_creation(self):
        """Test création du moniteur."""
        monitor = SweepMonitor(total_combinations=100)
        
        assert monitor.total == 100
        assert len(monitor.results) == 0
    
    def test_start(self):
        """Test démarrage."""
        monitor = SweepMonitor(total_combinations=100)
        monitor.start()
        
        assert monitor.stats.start_time is not None
    
    def test_update_increments_evaluated(self):
        """Test que update incrémente evaluated."""
        monitor = SweepMonitor(total_combinations=100)
        
        monitor.update(
            params={"x": 1},
            metrics={"sharpe_ratio": 1.0},
        )
        
        assert monitor.stats.evaluated == 1
        assert len(monitor.results) == 1
    
    def test_update_pruned(self):
        """Test update avec pruning."""
        monitor = SweepMonitor(total_combinations=100)
        
        monitor.update(
            params={"x": 1},
            metrics={},
            pruned=True,
        )
        
        assert monitor.stats.pruned == 1
        assert monitor.stats.evaluated == 0
    
    def test_update_error(self):
        """Test update avec erreur."""
        monitor = SweepMonitor(total_combinations=100)
        
        monitor.update(
            params={"x": 1},
            metrics={},
            error=True,
        )
        
        assert monitor.stats.errors == 1
        assert monitor.stats.evaluated == 0
    
    def test_top_results_tracking(self):
        """Test suivi des meilleurs résultats."""
        monitor = SweepMonitor(total_combinations=100, top_k=3)
        
        # Ajouter plusieurs résultats
        for i in range(5):
            monitor.update(
                params={"x": i},
                metrics={"sharpe_ratio": float(i)},
            )
        
        top = monitor.get_top_results("sharpe_ratio")
        
        assert len(top) == 3
        assert top[0].sharpe == 4.0  # Le meilleur
        assert top[1].sharpe == 3.0
    
    def test_get_best_result(self):
        """Test récupération du meilleur résultat."""
        monitor = SweepMonitor(total_combinations=100)
        
        monitor.update({"x": 1}, {"sharpe_ratio": 1.0})
        monitor.update({"x": 2}, {"sharpe_ratio": 2.0})
        monitor.update({"x": 3}, {"sharpe_ratio": 1.5})
        
        best = monitor.get_best_result("sharpe_ratio")
        
        assert best is not None
        assert best.params["x"] == 2
    
    def test_metric_history(self):
        """Test historique des métriques."""
        monitor = SweepMonitor(
            total_combinations=100,
            objectives=["sharpe_ratio"],
        )
        
        for i in range(5):
            monitor.update({"x": i}, {"sharpe_ratio": float(i)})
        
        history = monitor.get_metric_history("sharpe_ratio")
        
        assert len(history) == 5
        assert history == [0.0, 1.0, 2.0, 3.0, 4.0]
    
    def test_is_complete(self):
        """Test détection de completion."""
        monitor = SweepMonitor(total_combinations=3)
        
        assert not monitor.is_complete
        
        monitor.update({"x": 1}, {"sharpe_ratio": 1.0})
        monitor.update({"x": 2}, {"sharpe_ratio": 2.0})
        monitor.update({"x": 3}, {"sharpe_ratio": 3.0})
        
        assert monitor.is_complete
    
    def test_multiple_objectives(self):
        """Test avec plusieurs objectifs."""
        monitor = SweepMonitor(
            total_combinations=100,
            objectives=["sharpe_ratio", "max_drawdown"],
        )
        
        # Sharpe élevé, drawdown élevé
        monitor.update({"x": 1}, {"sharpe_ratio": 2.0, "max_drawdown": 0.30})
        # Sharpe moyen, drawdown faible
        monitor.update({"x": 2}, {"sharpe_ratio": 1.5, "max_drawdown": 0.10})
        
        best_sharpe = monitor.get_best_result("sharpe_ratio")
        best_dd = monitor.get_best_result("max_drawdown")
        
        # Meilleur sharpe = x:1
        assert best_sharpe.params["x"] == 1
        # Meilleur (plus bas) drawdown = x:2
        assert best_dd.params["x"] == 2


class TestIntegration:
    """Tests d'intégration des composants."""
    
    def test_monitor_with_sweep(self):
        """Test moniteur avec simulation de sweep."""
        monitor = SweepMonitor(
            total_combinations=10,
            objectives=["sharpe_ratio"],
        )
        
        monitor.start()
        
        # Simuler un sweep
        for i in range(10):
            monitor.update(
                params={"fast": 5 + i, "slow": 20 + i},
                metrics={"sharpe_ratio": 0.5 + i * 0.1},
                duration_ms=10.0,
            )
        
        assert monitor.is_complete
        assert monitor.stats.evaluated == 10
        assert monitor.stats.progress_percent == 100.0
        
        best = monitor.get_best_result("sharpe_ratio")
        assert best.params["fast"] == 14
        assert best.sharpe == pytest.approx(1.4, rel=0.01)
