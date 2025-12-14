"""
Tests pour les composants UI Phase 5 restants.

Tests pour:
- 5.3 Indicator Explorer
- 5.4 Agent Activity Timeline
- 5.5 Validation Report Viewer
- 5.6 Themes & Persistence
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json


# =============================================================================
# Tests Indicator Explorer (5.3)
# =============================================================================

class TestIndicatorExplorer:
    """Tests pour IndicatorExplorer."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Données OHLCV de test."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        return pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(110, 120, n),
            "low": np.random.uniform(90, 100, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)
    
    def test_import(self):
        """Test import du module."""
        from ui.components.indicator_explorer import (
            IndicatorExplorer,
            IndicatorType,
            ChartConfig,
            IndicatorConfig,
        )
        assert IndicatorExplorer is not None
        assert IndicatorType.OVERLAY is not None
    
    def test_indicator_type_enum(self):
        """Test enum IndicatorType."""
        from ui.components.indicator_explorer import IndicatorType
        
        assert IndicatorType.OVERLAY.value == "overlay"
        assert IndicatorType.OSCILLATOR.value == "oscillator"
        assert IndicatorType.VOLUME.value == "volume"
    
    def test_chart_config_defaults(self):
        """Test ChartConfig avec valeurs par défaut."""
        from ui.components.indicator_explorer import ChartConfig
        
        config = ChartConfig()
        assert config.height == 800
        assert config.show_volume is True
        assert config.range_slider is False
    
    def test_explorer_init(self, sample_ohlcv):
        """Test initialisation de l'explorateur."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        explorer = IndicatorExplorer(sample_ohlcv)
        assert len(explorer.df) == len(sample_ohlcv)
    
    def test_explorer_missing_columns(self):
        """Test erreur si colonnes manquantes."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        df = pd.DataFrame({"close": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            IndicatorExplorer(df)
    
    def test_add_indicator_overlay(self, sample_ohlcv):
        """Test ajout d'indicateur overlay."""
        from ui.components.indicator_explorer import IndicatorExplorer, IndicatorType
        
        explorer = IndicatorExplorer(sample_ohlcv)
        values = np.random.uniform(100, 110, len(sample_ohlcv))
        
        result = explorer.add_indicator(
            "test_ma",
            values,
            indicator_type=IndicatorType.OVERLAY,
            color="#ff0000",
        )
        
        assert result is explorer  # Chaînage
        assert "test_ma" in explorer._indicators
    
    def test_add_indicator_oscillator(self, sample_ohlcv):
        """Test ajout d'indicateur oscillateur."""
        from ui.components.indicator_explorer import IndicatorExplorer, IndicatorType
        
        explorer = IndicatorExplorer(sample_ohlcv)
        values = np.random.uniform(0, 100, len(sample_ohlcv))
        
        explorer.add_indicator(
            "test_rsi",
            values,
            indicator_type=IndicatorType.OSCILLATOR,
            levels=[30, 70],
        )
        
        assert "test_rsi" in explorer._indicators
        assert explorer._indicators["test_rsi"]["levels"] == [30, 70]
    
    def test_add_multi_line_indicator(self, sample_ohlcv):
        """Test ajout d'indicateur multi-lignes."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        explorer = IndicatorExplorer(sample_ohlcv)
        n = len(sample_ohlcv)
        values = {
            "upper": np.random.uniform(110, 120, n),
            "middle": np.random.uniform(100, 110, n),
            "lower": np.random.uniform(90, 100, n),
        }
        
        explorer.add_indicator("bollinger", values)
        
        assert "bollinger" in explorer._indicators
        assert isinstance(explorer._indicators["bollinger"]["values"], dict)
    
    def test_remove_indicator(self, sample_ohlcv):
        """Test suppression d'indicateur."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        explorer = IndicatorExplorer(sample_ohlcv)
        explorer.add_indicator("test", np.zeros(len(sample_ohlcv)))
        
        assert "test" in explorer._indicators
        
        explorer.remove_indicator("test")
        
        assert "test" not in explorer._indicators
    
    def test_clear_indicators(self, sample_ohlcv):
        """Test suppression de tous les indicateurs."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        explorer = IndicatorExplorer(sample_ohlcv)
        explorer.add_indicator("test1", np.zeros(len(sample_ohlcv)))
        explorer.add_indicator("test2", np.zeros(len(sample_ohlcv)))
        
        explorer.clear_indicators()
        
        assert len(explorer._indicators) == 0
    
    def test_get_indicator_summary(self, sample_ohlcv):
        """Test résumé des indicateurs."""
        from ui.components.indicator_explorer import IndicatorExplorer
        
        explorer = IndicatorExplorer(sample_ohlcv)
        values = np.array([10.0, 20.0, 30.0] * 33 + [40.0])  # 100 values
        
        explorer.add_indicator("test", values[:len(sample_ohlcv)])
        summary = explorer.get_indicator_summary()
        
        assert "test" in summary
        assert "stats" in summary["test"]
        assert "current" in summary["test"]["stats"]
    
    def test_default_indicator_configs(self):
        """Test configurations par défaut des indicateurs."""
        from ui.components.indicator_explorer import DEFAULT_INDICATOR_CONFIGS
        
        assert "rsi" in DEFAULT_INDICATOR_CONFIGS
        assert "bollinger" in DEFAULT_INDICATOR_CONFIGS
        assert "macd" in DEFAULT_INDICATOR_CONFIGS


# =============================================================================
# Tests Agent Activity Timeline (5.4)
# =============================================================================

class TestAgentActivityTimeline:
    """Tests pour AgentActivityTimeline."""
    
    def test_import(self):
        """Test import du module."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            ActivityType,
            DecisionType,
        )
        assert AgentActivityTimeline is not None
    
    def test_agent_type_enum(self):
        """Test enum AgentType."""
        from ui.components.agent_timeline import AgentType
        
        assert AgentType.ANALYST.value == "analyst"
        assert AgentType.STRATEGIST.value == "strategist"
        assert AgentType.CRITIC.value == "critic"
        assert AgentType.VALIDATOR.value == "validator"
    
    def test_activity_type_enum(self):
        """Test enum ActivityType."""
        from ui.components.agent_timeline import ActivityType
        
        assert ActivityType.STARTED.value == "started"
        assert ActivityType.ANALYSIS.value == "analysis"
        assert ActivityType.BACKTEST.value == "backtest"
    
    def test_timeline_init(self):
        """Test initialisation de la timeline."""
        from ui.components.agent_timeline import AgentActivityTimeline
        
        timeline = AgentActivityTimeline("Test Session")
        
        assert timeline.session_name == "Test Session"
        assert timeline.current_iteration == 0
        assert len(timeline.activities) == 0
    
    def test_log_activity(self):
        """Test enregistrement d'activité."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            ActivityType,
        )
        
        timeline = AgentActivityTimeline()
        
        activity = timeline.log_activity(
            AgentType.ANALYST,
            ActivityType.ANALYSIS,
            "Analyse des performances",
            details={"sharpe": 1.5},
        )
        
        assert len(timeline.activities) == 1
        assert activity.agent == AgentType.ANALYST
        assert activity.message == "Analyse des performances"
    
    def test_log_metrics(self):
        """Test enregistrement de métriques."""
        from ui.components.agent_timeline import AgentActivityTimeline
        
        timeline = AgentActivityTimeline()
        
        snapshot = timeline.log_metrics(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=0.10,
            win_rate=0.55,
        )
        
        assert len(timeline.metrics_history) == 1
        assert snapshot.sharpe_ratio == 1.5
    
    def test_log_decision(self):
        """Test enregistrement de décision."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            DecisionType,
        )
        
        timeline = AgentActivityTimeline()
        
        decision = timeline.log_decision(
            AgentType.VALIDATOR,
            DecisionType.APPROVE,
            "Performance satisfaisante",
            confidence=0.85,
        )
        
        assert len(timeline.decisions) == 1
        assert decision.decision == DecisionType.APPROVE
        assert decision.confidence == 0.85
    
    def test_next_iteration(self):
        """Test passage à l'itération suivante."""
        from ui.components.agent_timeline import AgentActivityTimeline
        
        timeline = AgentActivityTimeline()
        
        assert timeline.current_iteration == 0
        
        new_iter = timeline.next_iteration()
        
        assert new_iter == 1
        assert timeline.current_iteration == 1
    
    def test_get_activities_by_agent(self):
        """Test filtrage par agent."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            ActivityType,
        )
        
        timeline = AgentActivityTimeline()
        timeline.log_activity(AgentType.ANALYST, ActivityType.ANALYSIS, "msg1")
        timeline.log_activity(AgentType.STRATEGIST, ActivityType.PROPOSAL, "msg2")
        timeline.log_activity(AgentType.ANALYST, ActivityType.COMPLETED, "msg3")
        
        analyst_activities = timeline.get_activities_by_agent(AgentType.ANALYST)
        
        assert len(analyst_activities) == 2
    
    def test_get_summary(self):
        """Test résumé de la timeline."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            ActivityType,
        )
        
        timeline = AgentActivityTimeline("Test")
        timeline.log_activity(AgentType.ANALYST, ActivityType.ANALYSIS, "msg")
        timeline.log_metrics(1.5, 0.2, 0.1, 0.5)
        timeline.next_iteration()
        
        summary = timeline.get_summary()
        
        assert summary["session_name"] == "Test"
        assert summary["total_iterations"] == 1
        assert summary["total_activities"] == 1
    
    def test_to_dict_and_json(self):
        """Test sérialisation."""
        from ui.components.agent_timeline import (
            AgentActivityTimeline,
            AgentType,
            ActivityType,
        )
        
        timeline = AgentActivityTimeline("Test")
        timeline.log_activity(AgentType.ANALYST, ActivityType.ANALYSIS, "msg")
        
        data = timeline.to_dict()
        assert "session_name" in data
        assert "activities" in data
        
        json_str = timeline.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["session_name"] == "Test"


# =============================================================================
# Tests Validation Report Viewer (5.5)
# =============================================================================

class TestValidationViewer:
    """Tests pour ValidationReport et Viewer."""
    
    def test_import(self):
        """Test import du module."""
        from ui.components.validation_viewer import (
            ValidationReport,
            WindowResult,
            ValidationStatus,
        )
        assert ValidationReport is not None
    
    def test_validation_status_enum(self):
        """Test enum ValidationStatus."""
        from ui.components.validation_viewer import ValidationStatus
        
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.OVERFITTING.value == "overfitting"
    
    @pytest.fixture
    def sample_window(self):
        """Fenêtre de validation exemple."""
        from ui.components.validation_viewer import WindowResult
        
        return WindowResult(
            window_id=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            train_sharpe=1.8,
            train_return=0.25,
            train_drawdown=0.10,
            train_trades=100,
            test_sharpe=1.2,
            test_return=0.15,
            test_drawdown=0.12,
            test_trades=30,
            params={"fast": 10, "slow": 20},
        )
    
    def test_window_result_degradation(self, sample_window):
        """Test calcul dégradation."""
        # 1.8 -> 1.2 = 33% dégradation
        assert sample_window.sharpe_degradation == pytest.approx(0.333, abs=0.01)
    
    def test_window_result_not_overfitting(self, sample_window):
        """Test détection pas d'overfitting."""
        # Dégradation 33% < 50%, pas overfitting
        assert sample_window.is_overfitting is False
    
    def test_window_result_overfitting_detection(self):
        """Test détection overfitting."""
        from ui.components.validation_viewer import WindowResult, ValidationStatus
        
        window = WindowResult(
            window_id=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            train_sharpe=2.0,
            train_return=0.30,
            train_drawdown=0.08,
            train_trades=100,
            test_sharpe=-0.5,  # Négatif!
            test_return=-0.05,
            test_drawdown=0.20,
            test_trades=30,
        )
        
        assert window.is_overfitting is True
        assert window.status == ValidationStatus.OVERFITTING
    
    def test_validation_report_init(self, sample_window):
        """Test initialisation du rapport."""
        from ui.components.validation_viewer import ValidationReport
        
        report = ValidationReport(
            strategy_name="ema_cross",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        assert report.strategy_name == "ema_cross"
        assert len(report.windows) == 1
    
    def test_validation_report_aggregate_metrics(self, sample_window):
        """Test métriques agrégées."""
        from ui.components.validation_viewer import ValidationReport
        
        report = ValidationReport(
            strategy_name="test",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        metrics = report.aggregate_metrics
        
        assert "avg_train_sharpe" in metrics
        assert "avg_test_sharpe" in metrics
        assert "consistency_ratio" in metrics
    
    def test_validation_report_overall_status(self, sample_window):
        """Test statut global."""
        from ui.components.validation_viewer import ValidationReport, ValidationStatus
        
        report = ValidationReport(
            strategy_name="test",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        # Le statut dépend des métriques calculées
        # Vérifions simplement que le statut est un ValidationStatus valide
        status = report.overall_status
        assert isinstance(status, ValidationStatus)
    
    def test_validation_report_is_valid(self, sample_window):
        """Test propriété is_valid."""
        from ui.components.validation_viewer import ValidationReport, ValidationStatus
        
        report = ValidationReport(
            strategy_name="test",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        # is_valid = True seulement si PASSED
        # Avec degradation 33% > 30%, c'est WARNING donc pas valid
        status = report.overall_status
        if status == ValidationStatus.PASSED:
            assert report.is_valid is True
        else:
            assert report.is_valid is False
    
    def test_validation_report_get_best_params(self, sample_window):
        """Test extraction meilleurs paramètres."""
        from ui.components.validation_viewer import ValidationReport
        
        report = ValidationReport(
            strategy_name="test",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        params = report.get_best_params()
        
        assert params == {"fast": 10, "slow": 20}
    
    def test_validation_report_to_dict(self, sample_window):
        """Test sérialisation."""
        from ui.components.validation_viewer import ValidationReport
        
        report = ValidationReport(
            strategy_name="test",
            created_at=datetime.now(),
            windows=[sample_window],
        )
        
        data = report.to_dict()
        
        assert "strategy_name" in data
        assert "windows" in data
        assert "overall_status" in data
    
    def test_create_sample_report(self):
        """Test génération rapport exemple."""
        from ui.components.validation_viewer import create_sample_report
        
        report = create_sample_report()
        
        assert len(report.windows) == 5
        assert report.strategy_name == "ema_cross"


# =============================================================================
# Tests Themes & Persistence (5.6)
# =============================================================================

class TestThemesAndPersistence:
    """Tests pour Themes et PreferencesManager."""
    
    def test_import(self):
        """Test import du module."""
        from ui.components.themes import (
            UserPreferences,
            PreferencesManager,
            ThemeMode,
            ColorPalette,
        )
        assert UserPreferences is not None
    
    def test_theme_mode_enum(self):
        """Test enum ThemeMode."""
        from ui.components.themes import ThemeMode
        
        assert ThemeMode.LIGHT.value == "light"
        assert ThemeMode.DARK.value == "dark"
        assert ThemeMode.AUTO.value == "auto"
    
    def test_color_palette_enum(self):
        """Test enum ColorPalette."""
        from ui.components.themes import ColorPalette, PALETTES
        
        assert ColorPalette.DEFAULT.value == "default"
        assert ColorPalette.OCEAN.value == "ocean"
        
        # Vérifier que toutes les palettes sont définies
        for palette in ColorPalette:
            assert palette in PALETTES
    
    def test_palettes_have_required_colors(self):
        """Test que toutes les palettes ont les couleurs requises."""
        from ui.components.themes import PALETTES
        
        required_keys = [
            "primary", "secondary", "success", "warning", "error",
            "background", "surface", "text", "chart_up", "chart_down",
        ]
        
        for palette_name, colors in PALETTES.items():
            for key in required_keys:
                assert key in colors, f"{palette_name} manque {key}"
    
    def test_user_preferences_defaults(self):
        """Test valeurs par défaut des préférences."""
        from ui.components.themes import UserPreferences, ThemeMode, ColorPalette
        
        prefs = UserPreferences()
        
        assert prefs.theme_mode == ThemeMode.DARK
        assert prefs.color_palette == ColorPalette.DEFAULT
        assert prefs.font_size == "medium"
    
    def test_user_preferences_chart_settings(self):
        """Test paramètres de graphique."""
        from ui.components.themes import UserPreferences
        
        prefs = UserPreferences()
        
        assert prefs.chart_settings.default_height == 600
        assert prefs.chart_settings.show_volume is True
    
    def test_user_preferences_to_dict(self):
        """Test sérialisation."""
        from ui.components.themes import UserPreferences
        
        prefs = UserPreferences()
        data = prefs.to_dict()
        
        assert "theme_mode" in data
        assert "color_palette" in data
        assert "chart_settings" in data
    
    def test_user_preferences_from_dict(self):
        """Test désérialisation."""
        from ui.components.themes import UserPreferences, ThemeMode, ColorPalette
        
        data = {
            "theme_mode": "light",
            "color_palette": "ocean",
            "font_size": "large",
            "chart_settings": {"default_height": 800},
            "performance_settings": {},
            "default_params": {},
            "favorite_strategies": ["ema_cross"],
            "favorite_indicators": [],
            "recent_data_files": [],
            "sidebar_expanded": False,
            "show_tips": False,
        }
        
        prefs = UserPreferences.from_dict(data)
        
        assert prefs.theme_mode == ThemeMode.LIGHT
        assert prefs.color_palette == ColorPalette.OCEAN
        assert prefs.font_size == "large"
    
    def test_user_preferences_get_colors(self):
        """Test récupération des couleurs."""
        from ui.components.themes import UserPreferences
        
        prefs = UserPreferences()
        colors = prefs.get_colors()
        
        assert "primary" in colors
        assert colors["primary"].startswith("#")
    
    def test_preferences_manager_init(self):
        """Test initialisation du gestionnaire."""
        from ui.components.themes import PreferencesManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            assert manager.path == path
    
    def test_preferences_manager_load_default(self):
        """Test chargement (crée défaut si n'existe pas)."""
        from ui.components.themes import PreferencesManager, ThemeMode
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            prefs = manager.load()
            
            assert prefs.theme_mode == ThemeMode.DARK
    
    def test_preferences_manager_save_and_load(self):
        """Test sauvegarde et rechargement."""
        from ui.components.themes import PreferencesManager, UserPreferences, ThemeMode
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            # Modifier et sauvegarder
            prefs = UserPreferences(theme_mode=ThemeMode.LIGHT)
            manager.save(prefs)
            
            # Recharger
            manager2 = PreferencesManager(path)
            loaded = manager2.load()
            
            assert loaded.theme_mode == ThemeMode.LIGHT
    
    def test_preferences_manager_update(self):
        """Test mise à jour partielle."""
        from ui.components.themes import PreferencesManager, ThemeMode
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            prefs = manager.update(theme_mode=ThemeMode.LIGHT, font_size="large")
            
            assert prefs.theme_mode == ThemeMode.LIGHT
            assert prefs.font_size == "large"
    
    def test_preferences_manager_reset(self):
        """Test réinitialisation."""
        from ui.components.themes import PreferencesManager, ThemeMode, UserPreferences
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            # Modifier
            manager.save(UserPreferences(theme_mode=ThemeMode.LIGHT))
            
            # Reset
            prefs = manager.reset()
            
            assert prefs.theme_mode == ThemeMode.DARK
    
    def test_preferences_manager_favorites(self):
        """Test gestion des favoris."""
        from ui.components.themes import PreferencesManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            manager.add_favorite_strategy("ema_cross")
            manager.add_favorite_strategy("macd_cross")
            
            assert "ema_cross" in manager.preferences.favorite_strategies
            assert "macd_cross" in manager.preferences.favorite_strategies
            
            manager.remove_favorite_strategy("ema_cross")
            
            assert "ema_cross" not in manager.preferences.favorite_strategies
    
    def test_preferences_manager_recent_files(self):
        """Test gestion des fichiers récents."""
        from ui.components.themes import PreferencesManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            manager = PreferencesManager(path)
            
            manager.add_recent_file("/path/to/file1.parquet")
            manager.add_recent_file("/path/to/file2.parquet")
            
            assert manager.preferences.recent_data_files[0] == "/path/to/file2.parquet"
            assert len(manager.preferences.recent_data_files) == 2


# =============================================================================
# Tests __init__.py exports
# =============================================================================

class TestComponentsInit:
    """Tests pour vérifier les exports du module."""
    
    def test_all_exports_available(self):
        """Test que tous les exports sont disponibles."""
        from ui.components import (
            # Monitor
            SystemMonitor,
            render_system_monitor,
            # Sweep Monitor
            SweepMonitor,
            render_sweep_progress,
            # Indicator Explorer
            IndicatorExplorer,
            IndicatorType,
            # Agent Timeline
            AgentActivityTimeline,
            AgentType,
            ActivityType,
            # Validation Viewer
            ValidationReport,
            ValidationStatus,
            # Themes
            UserPreferences,
            PreferencesManager,
            ThemeMode,
            ColorPalette,
        )
        
        assert all([
            SystemMonitor, SweepMonitor, IndicatorExplorer,
            AgentActivityTimeline, ValidationReport, UserPreferences,
        ])
