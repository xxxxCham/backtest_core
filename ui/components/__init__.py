"""
Backtest Core - UI Components
=============================

Composants réutilisables pour l'interface Streamlit.

Phase 5 - UI/UX Avancée:
- 5.1 System Monitor (monitoring CPU/RAM/GPU)
- 5.2 Sweep Monitor (progress tracking)
- 5.3 Indicator Explorer (visualisation indicateurs)
- 5.4 Agent Timeline (suivi agents LLM)
- 5.5 Validation Viewer (rapports walk-forward)
- 5.6 Themes & Persistence (thèmes et préférences)
"""

from .monitor import SystemMonitor, render_system_monitor, render_mini_monitor
from .sweep_monitor import SweepMonitor, render_sweep_progress, render_sweep_summary
from .indicator_explorer import (
    IndicatorExplorer,
    IndicatorType,
    ChartConfig,
    render_indicator_explorer,
    render_quick_indicator_chart,
)
from .agent_timeline import (
    AgentActivityTimeline,
    AgentActivity,
    AgentType,
    ActivityType,
    DecisionType,
    render_agent_timeline,
    render_mini_timeline,
)
from .validation_viewer import (
    ValidationReport,
    WindowResult,
    ValidationStatus,
    render_validation_report,
    render_validation_summary_card,
)
from .themes import (
    UserPreferences,
    PreferencesManager,
    ThemeMode,
    ColorPalette,
    get_preferences,
    get_preferences_manager,
    apply_theme,
    render_theme_settings,
    render_chart_settings,
    render_full_settings_page,
)
from .model_selector import (
    FALLBACK_LLM_MODELS,
    RECOMMENDED_FOR_ANALYSIS,
    RECOMMENDED_FOR_STRATEGY,
    RECOMMENDED_FOR_CRITICISM,
    RECOMMENDED_FOR_FAST,
    get_available_models_for_ui,
    get_model_info,
    render_model_selector,
)
from .charts import (
    render_equity_and_drawdown,
    render_ohlcv_with_trades,
    render_ohlcv_with_indicators,
    render_equity_curve,
    render_comparison_chart,
)
from .thinking_viewer import (
    ThinkingStreamViewer,
    ThoughtEntry,
    ThoughtCategory,
    render_thinking_stream,
)

__all__ = [
    # Monitor
    "SystemMonitor",
    "render_system_monitor",
    "render_mini_monitor",
    # Sweep Monitor
    "SweepMonitor",
    "render_sweep_progress",
    "render_sweep_summary",
    # Indicator Explorer
    "IndicatorExplorer",
    "IndicatorType",
    "ChartConfig",
    "render_indicator_explorer",
    "render_quick_indicator_chart",
    # Agent Timeline
    "AgentActivityTimeline",
    "AgentActivity",
    "AgentType",
    "ActivityType",
    "DecisionType",
    "render_agent_timeline",
    "render_mini_timeline",
    # Validation Viewer
    "ValidationReport",
    "WindowResult",
    "ValidationStatus",
    "render_validation_report",
    "render_validation_summary_card",
    # Themes
    "UserPreferences",
    "PreferencesManager",
    "ThemeMode",
    "ColorPalette",
    "get_preferences",
    "get_preferences_manager",
    "apply_theme",
    "render_theme_settings",
    "render_chart_settings",
    "render_full_settings_page",
    # Model Selector
    "FALLBACK_LLM_MODELS",
    "RECOMMENDED_FOR_ANALYSIS",
    "RECOMMENDED_FOR_STRATEGY",
    "RECOMMENDED_FOR_CRITICISM",
    "RECOMMENDED_FOR_FAST",
    "get_available_models_for_ui",
    "get_model_info",
    "render_model_selector",
    # Charts
    "render_equity_and_drawdown",
    "render_ohlcv_with_trades",
    "render_ohlcv_with_indicators",
    "render_equity_curve",
    "render_comparison_chart",
    # Thinking Viewer
    "ThinkingStreamViewer",
    "ThoughtEntry",
    "ThoughtCategory",
    "render_thinking_stream",
]
