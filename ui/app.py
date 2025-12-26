"""
Backtest Core - Streamlit Application v2
========================================

Interface utilisateur robuste avec:
- Validation des paramètres avec contraintes
Save failed: st.session_state.versioned_preset_version cannot be modified after the widget with key versioned_preset_version is instantiated.Save failed: st.session_state.versioned_preset_version cannot be modified after the widget with key versioned_preset_version is instantiated.- Feedback utilisateur clair (success/error/warning)
- Gestion d'erreurs complète
- Visualisation améliorée des résultats

Lancer avec: streamlit run ui/app.py
"""

import logging
import math
import re
import statistics
import sys
import threading
import time
import traceback
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports du moteur (backend)
IMPORT_ERROR = ""  # Initialisé avant le try/except
try:
    from backtest.engine import BacktestEngine, RunResult
    from data.loader import discover_available_data, load_ohlcv
    from indicators.registry import calculate_indicator
    from strategies.base import get_strategy, list_strategies
    from strategies.indicators_mapping import get_strategy_info
    from utils.parameters import (
        ParameterSpec,
        compute_search_space_stats,
        list_strategy_versions,
        load_strategy_version,
        resolve_latest_version,
        save_versioned_preset,
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Imports agents LLM (optionnels)
LLM_AVAILABLE = False
LLM_IMPORT_ERROR = ""
try:
    from agents.autonomous_strategist import AutonomousStrategist
    from agents.integration import (
        create_optimizer_from_engine,
        create_orchestrator_with_backtest,
        get_strategy_param_bounds,
        get_strategy_param_space,
    )
    from agents.llm_client import LLMConfig, LLMProvider, create_llm_client
    from agents.model_config import (
        KNOWN_MODELS,
        ModelCategory,
        ModelInfo,
        RoleModelConfig,
        get_global_model_config,
        get_models_by_category,
        list_available_models,
        set_global_model_config,
    )
    from agents.ollama_manager import (
        ensure_ollama_running,
        is_ollama_available,
    )
    from agents.orchestration_logger import OrchestrationLogger, generate_session_id
    from ui.components.agent_timeline import (
        ActivityType,
        AgentActivity,
        AgentActivityTimeline,
        AgentType,
        render_agent_timeline,
        render_mini_timeline,
    )
    from ui.components.model_selector import (
        RECOMMENDED_FOR_STRATEGY,
        get_available_models_for_ui,
        get_model_info,
    )
    from ui.components.monitor import render_mini_monitor
    from ui.deep_trace_viewer import render_deep_trace_viewer  # Pour onglet avancé mode LLM
    from ui.orchestration_viewer import (
        LiveOrchestrationViewer,
        render_full_orchestration_viewer,
        render_live_orchestration_panel,
        render_orchestration_logs,
        render_orchestration_summary_table,
    )
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_IMPORT_ERROR = str(e)

# Import observabilité (toujours disponible)
# Import composants UI (toujours disponibles)
from ui.components.charts import (
    render_comparison_chart,
    render_ohlcv_with_trades,
    render_ohlcv_with_trades_and_indicators,
    render_strategy_param_diagram,
)
from utils.observability import (
    generate_run_id,
    get_obs_logger,
    init_logging,
    is_debug_enabled,
    set_log_level,
)
from utils.run_tracker import RunSignature, get_global_tracker

# Initialiser le logging au démarrage de l'UI
init_logging()


# ============================================================================
# LOG TAP: BEST PNL SEEN IN CONSOLE LOGS
# ============================================================================
_BEST_PNL_TRACKER = None


class _BestPnlTracker(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.best_pnl: Optional[float] = None
        self.best_run_id: Optional[str] = None
        self._lock = threading.Lock()
        self._pnl_pattern = re.compile(
            r"\bpnl\s*=\s*[^0-9-+]*([-+]?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "backtest.engine":
            return
        msg = record.getMessage()
        if "pnl" not in msg.lower():
            return
        match = self._pnl_pattern.search(msg)
        if not match:
            return
        try:
            pnl = float(match.group(1))
        except ValueError:
            return
        with self._lock:
            if self.best_pnl is None or pnl > self.best_pnl:
                self.best_pnl = pnl
                self.best_run_id = getattr(record, "run_id", None)

    def get_best(self) -> Tuple[Optional[float], Optional[str]]:
        with self._lock:
            return self.best_pnl, self.best_run_id


def _install_best_pnl_tracker() -> _BestPnlTracker:
    global _BEST_PNL_TRACKER
    if _BEST_PNL_TRACKER is not None:
        return _BEST_PNL_TRACKER
    logger = logging.getLogger("backtest")
    for handler in logger.handlers:
        if isinstance(handler, _BestPnlTracker):
            _BEST_PNL_TRACKER = handler
            return handler
    tracker = _BestPnlTracker()
    logger.addHandler(tracker)
    _BEST_PNL_TRACKER = tracker
    return tracker


_BEST_PNL_TRACKER = _install_best_pnl_tracker()


# ============================================================================
# CONFIGURATION PAGE
# ============================================================================
st.set_page_config(
    page_title="Backtest Core",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTES ET CONTRAINTES
# ============================================================================

# Contraintes des paramètres (min, max, step, description)
# Plages étendues pour permettre plus de combinaisons de test
PARAM_CONSTRAINTS = {
    # Bollinger ATR Strategy
    "bb_period": {
        "min": 2, "max": 200, "step": 1, "default": 20,
        "description": "Période des Bollinger Bands (2-200)"
    },
    "bb_std": {
        "min": 0.5, "max": 5.0, "step": 0.1, "default": 2.0,
        "description": "Écart-type des bandes (0.5-5.0)"
    },
    "bb_window": {
        "min": 10, "max": 50, "step": 1, "default": 20,
        "description": "Periode Bollinger (10-50)"
    },
    "ma_window": {
        "min": 5, "max": 30, "step": 1, "default": 10,
        "description": "Periode MA (5-30)"
    },
    "trailing_pct": {
        "min": 0.5, "max": 1.0, "step": 0.05, "default": 0.8,
        "description": "Trailing stop (0.5-1.0)"
    },
    "short_stop_pct": {
        "min": 0.1, "max": 0.5, "step": 0.01, "default": 0.37,
        "description": "Stop loss short (0.1-0.5)"
    },
    "atr_period": {
        "min": 2, "max": 100, "step": 1, "default": 14,
        "description": "Période ATR (2-100)"
    },
    "atr_percentile": {
        "min": 10, "max": 60, "step": 1, "default": 30,
        "description": "Percentile ATR (10-60)"
    },
    "entry_z": {
        "min": 0.5, "max": 5.0, "step": 0.1, "default": 2.0,
        "description": "Z-score d'entrée (0.5-5.0)"
    },
    "k_sl": {
        "min": 0.1, "max": 10.0, "step": 0.1, "default": 1.5,
        "description": "Multiplicateur stop-loss (0.1-10.0)"
    },
    # Commun
    "leverage": {
        "min": 1, "max": 100, "step": 1, "default": 1,
        "description": "Levier de trading (1-100)"
    },
    # EMA Cross / MA Crossover Strategy
    "fast_period": {
        "min": 2, "max": 200, "step": 1, "default": 12,
        "description": "Période MA rapide (2-200)"
    },
    "slow_period": {
        "min": 2, "max": 500, "step": 1, "default": 26,
        "description": "Période MA lente (2-500)"
    },
    "ema_fast": {
        "min": 10, "max": 50, "step": 1, "default": 20,
        "description": "Période EMA rapide (10-50)"
    },
    "ema_slow": {
        "min": 30, "max": 100, "step": 1, "default": 50,
        "description": "Période EMA lente (30-100)"
    },
    # MACD Cross Strategy
    "signal_period": {
        "min": 2, "max": 50, "step": 1, "default": 9,
        "description": "Période ligne signal MACD (2-50)"
    },
    # RSI Reversal Strategy
    "rsi_period": {
        "min": 2, "max": 100, "step": 1, "default": 14,
        "description": "Période RSI (2-100)"
    },
    "oversold_level": {
        "min": 1, "max": 49, "step": 1, "default": 30,
        "description": "Seuil survente RSI (1-49)"
    },
    "overbought_level": {
        "min": 51, "max": 99, "step": 1, "default": 70,
        "description": "Seuil surachat RSI (51-99)"
    },
    # ATR Channel Strategy
    "atr_mult": {
        "min": 0.1, "max": 10.0, "step": 0.1, "default": 2.0,
        "description": "Multiplicateur ATR pour canal (0.1-10.0)"
    },
    # EMA Stochastic Scalp Strategy
    "fast_ema": {
        "min": 2, "max": 200, "step": 1, "default": 50,
        "description": "Période EMA rapide scalp (2-200)"
    },
    "slow_ema": {
        "min": 2, "max": 500, "step": 1, "default": 100,
        "description": "Période EMA lente scalp (2-500)"
    },
    "stoch_k": {
        "min": 2, "max": 100, "step": 1, "default": 14,
        "description": "Période Stochastic %K (2-100)"
    },
    "stoch_d": {
        "min": 1, "max": 50, "step": 1, "default": 3,
        "description": "Période Stochastic %D (1-50)"
    },
    "stoch_oversold": {
        "min": 1, "max": 49, "step": 1, "default": 20,
        "description": "Seuil survente Stochastic (1-49)"
    },
    "stoch_overbought": {
        "min": 51, "max": 99, "step": 1, "default": 80,
        "description": "Seuil surachat Stochastic (51-99)"
    },
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def generate_strategies_table() -> str:
    """
    Génère dynamiquement le tableau markdown des stratégies disponibles.

    Synchronise automatiquement avec le registre des stratégies pour éviter
    toute divergence entre la sidebar et la page principale.

    Returns:
        str: Tableau markdown formaté avec toutes les stratégies enregistrées
    """
    # Récupérer toutes les stratégies depuis le registre
    available = list_strategies()

    # Métadonnées des stratégies (défini dans la section stratégie de la sidebar)
    # On utilise les mêmes dictionnaires pour assurer la cohérence
    display_names = {
        "bollinger_atr": "Bollinger + ATR",
        "ema_cross": "EMA Crossover",
        "macd_cross": "MACD Crossover",
        "rsi_reversal": "RSI Reversal",
        "atr_channel": "ATR Channel",
        "ma_crossover": "MA Crossover",
        "ema_stochastic_scalp": "EMA + Stochastic",
        "bollinger_dual": "Bollinger Dual",
        "rsi_trend_filtered": "RSI Trend Filtered",
    }

    types = {
        "bollinger_atr": "Mean Rev.",
        "ema_cross": "Trend",
        "macd_cross": "Momentum",
        "rsi_reversal": "Mean Rev.",
        "atr_channel": "Breakout",
        "ma_crossover": "Trend",
        "ema_stochastic_scalp": "Scalping",
        "bollinger_dual": "Mean Rev.",
        "rsi_trend_filtered": "Mean Rev.",
    }

    descriptions = {
        "bollinger_atr": "Bandes + filtre volatilité",
        "ema_cross": "Golden/Death cross EMAs",
        "macd_cross": "MACD vs Signal line",
        "rsi_reversal": "Survente/Surachat RSI",
        "atr_channel": "Breakout EMA+ATR",
        "ma_crossover": "Croisement SMA",
        "ema_stochastic_scalp": "Scalping EMA + Stoch",
        "bollinger_dual": "Double condition BB + MA",
        "rsi_trend_filtered": "RSI filtré tendance",
    }

    # Construire le tableau markdown
    table_lines = [
        "### Stratégies Disponibles",
        "",
        "| Stratégie | Type | Description |",
        "|-----------|------|-------------|",
    ]

    # Ajouter chaque stratégie enregistrée
    for strat_key in sorted(available):
        name = display_names.get(strat_key, strat_key)
        stype = types.get(strat_key, "Autre")
        desc = descriptions.get(strat_key, "Stratégie personnalisée")
        table_lines.append(f"| **{name}** | {stype} | {desc} |")

    return "\n".join(table_lines)


class ProgressMonitor:
    """
    Moniteur de progression en temps réel pour les backtests.

    Calcule la vitesse d'exécution et estime le temps restant en utilisant
    une moyenne glissante sur les 3 dernières secondes.
    """

    def __init__(self, total_runs: int):
        """
        Initialise le moniteur.

        Args:
            total_runs: Nombre total d'itérations à effectuer
        """
        self.total_runs = total_runs
        self.runs_completed = 0
        self.start_time = time.perf_counter()
        # Historique : [(timestamp, runs_completed), ...]
        self.history = deque(maxlen=3)  # 3 derniers points de mesure
        self.last_update_time = self.start_time

    def update(self, runs_completed: int) -> Dict[str, Any]:
        """
        Met à jour le moniteur avec le nombre d'itérations complétées.

        Args:
            runs_completed: Nombre d'itérations complétées

        Returns:
            Dict avec les métriques calculées
        """
        self.runs_completed = runs_completed
        current_time = time.perf_counter()

        # Ajouter (timestamp, runs_completed) à l'historique
        self.history.append((current_time, runs_completed))

        # Calculer la vitesse (moyenne glissante sur l'historique)
        if len(self.history) >= 2:
            # Temps et runs entre premier et dernier point de l'historique
            time_span = self.history[-1][0] - self.history[0][0]
            runs_in_span = self.history[-1][1] - self.history[0][1]

            if time_span > 0 and runs_in_span > 0:
                iteration_speed_per_sec = runs_in_span / time_span
                iteration_speed_per_2sec = iteration_speed_per_sec * 2
            else:
                iteration_speed_per_sec = 0
                iteration_speed_per_2sec = 0
        else:
            iteration_speed_per_sec = 0
            iteration_speed_per_2sec = 0

        # Temps écoulé total
        elapsed_time = current_time - self.start_time

        # Estimation du temps restant
        remaining_runs = self.total_runs - runs_completed
        if iteration_speed_per_sec > 0 and remaining_runs > 0:
            time_remaining_sec = remaining_runs / iteration_speed_per_sec
        else:
            time_remaining_sec = 0

        # Progression
        progress = runs_completed / self.total_runs if self.total_runs > 0 else 0

        self.last_update_time = current_time

        return {
            "progress": progress,
            "runs_completed": runs_completed,
            "total_runs": self.total_runs,
            "speed_per_2sec": iteration_speed_per_2sec,
            "speed_per_sec": iteration_speed_per_sec,
            "elapsed_time_sec": elapsed_time,
            "time_remaining_sec": time_remaining_sec,
        }

    def format_time(self, seconds: float) -> str:
        """
        Formate un temps en secondes en format lisible.

        Args:
            seconds: Temps en secondes

        Returns:
            String formaté (ex: "2h 15m 30s")
        """
        if seconds <= 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)


def render_progress_monitor(monitor: ProgressMonitor, placeholder) -> None:
    """
    Affiche le moniteur de progression en temps réel.

    Args:
        monitor: Instance du ProgressMonitor
        placeholder: Placeholder Streamlit pour l'affichage
    """
    metrics = monitor.update(monitor.runs_completed)

    with placeholder.container():
        # Barre de progression
        st.progress(metrics["progress"])

        # Métriques en colonnes
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Progression",
                f"{metrics['runs_completed']}/{metrics['total_runs']}",
                f"{metrics['progress']*100:.1f}%"
            )

        with col2:
            st.metric(
                "Vitesse",
                f"{metrics['speed_per_sec']:.2f} runs/s",
                f"{metrics['speed_per_2sec']:.1f} runs/2s"
            )

        with col3:
            elapsed_str = monitor.format_time(metrics['elapsed_time_sec'])
            st.metric(
                "Temps écoulé",
                elapsed_str
            )

        with col4:
            remaining_str = monitor.format_time(metrics['time_remaining_sec'])
            st.metric(
                "Temps restant",
                remaining_str
            )


def show_status(status_type: str, message: str, details: Optional[str] = None):
    """Affiche un message de statut formaté."""
    if status_type == "success":
        st.success(f"✅ {message}")
    elif status_type == "error":
        st.error(f"❌ {message}")
        if details:
            with st.expander("Détails de l'erreur"):
                st.code(details)
    elif status_type == "warning":
        st.warning(f"⚠️ {message}")
    elif status_type == "info":
        st.info(f"ℹ️ {message}")


def validate_param(name: str, value: Any) -> Tuple[bool, str]:
    """
    Valide un paramètre selon ses contraintes.

    Returns:
        (is_valid, error_message)
    """
    if name not in PARAM_CONSTRAINTS:
        return True, ""

    constraints = PARAM_CONSTRAINTS[name]

    if value < constraints["min"]:
        return False, f"{name} doit être ≥ {constraints['min']}"

    if value > constraints["max"]:
        return False, f"{name} doit être ≤ {constraints['max']}"

    return True, ""


def validate_all_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valide tous les paramètres et retourne les erreurs."""
    errors = []

    for name, value in params.items():
        is_valid, error = validate_param(name, value)
        if not is_valid:
            errors.append(error)

    # Validations croisées
    if "fast_period" in params and "slow_period" in params:
        if params["fast_period"] >= params["slow_period"]:
            errors.append("fast_period doit être < slow_period")

    return len(errors) == 0, errors


def apply_versioned_preset(preset: Any, strategy_key: str) -> None:
    """Apply preset default values into Streamlit session state."""
    try:
        values = preset.get_default_values()
    except Exception:
        values = {}

    for name, value in values.items():
        st.session_state[f"{strategy_key}_{name}"] = value

    if "leverage" in values:
        st.session_state["trading_leverage"] = values["leverage"]


def create_param_range_selector(
    name: str,
    key_prefix: str = "",
    mode: str = "single",  # "single" ou "range"
    spec: Optional[ParameterSpec] = None,
) -> Any:
    """
    Crée un sélecteur de paramètre avec contrôle min/max/step individuel.

    Args:
        name: Nom du paramètre
        key_prefix: Préfixe pour les clés Streamlit
        mode: "single" = une valeur, "range" = plage min/max/step

    Returns:
        Valeur unique ou dict {"min", "max", "step"} selon le mode
    """
    constraints: Dict[str, Any] = {}
    is_int = False

    if spec is not None:
        spec_type = spec.param_type
        is_int = spec_type == "int" or spec_type is int
        step = spec.step
        if step is None:
            range_size = float(spec.max_val) - float(spec.min_val)
            if is_int:
                step = max(1, int(range_size / 10))
            else:
                step = range_size / 10 if range_size > 0 else 0.1
        if is_int:
            step = max(1, int(round(step)))
        constraints = {
            "min": spec.min_val,
            "max": spec.max_val,
            "step": step,
            "default": spec.default,
            "description": spec.description,
            "type": "int" if is_int else "float",
        }
    else:
        if name not in PARAM_CONSTRAINTS:
            st.sidebar.warning(f"Paramètre {name} sans contraintes définies")
            return None
        constraints = PARAM_CONSTRAINTS[name]
        step = constraints.get("step", 1)
        is_int = constraints.get("type") == "int"
        if not is_int:
            try:
                is_int = float(step).is_integer()
            except (TypeError, ValueError):
                is_int = False

    unique_key = f"{key_prefix}_{name}"

    if mode == "single":
        # Mode simple: un seul slider
        if is_int:
            return st.sidebar.slider(
                name,
                min_value=int(constraints["min"]),
                max_value=int(constraints["max"]),
                value=int(constraints["default"]),
                step=int(constraints["step"]),
                help=constraints["description"],
                key=unique_key
            )
        else:
            return st.sidebar.slider(
                name,
                min_value=float(constraints["min"]),
                max_value=float(constraints["max"]),
                value=float(constraints["default"]),
                step=float(constraints["step"]),
                help=constraints["description"],
                key=unique_key
            )
    else:
        # Mode range: sélection min/max/step pour grille
        with st.sidebar.expander(f"📊 {name}", expanded=False):
            st.caption(constraints["description"])

            col1, col2 = st.columns(2)

            if is_int:
                with col1:
                    param_min = st.number_input(
                        "Min",
                        min_value=int(constraints["min"]),
                        max_value=int(constraints["max"]),
                        value=int(constraints["min"]),
                        step=1,
                        key=f"{unique_key}_min"
                    )
                with col2:
                    param_max = st.number_input(
                        "Max",
                        min_value=int(constraints["min"]),
                        max_value=int(constraints["max"]),
                        value=int(constraints["max"]),
                        step=1,
                        key=f"{unique_key}_max"
                    )
                param_step = st.number_input(
                    "Step",
                    min_value=1,
                    max_value=max(1, (int(constraints["max"]) - int(constraints["min"])) // 2),
                    value=int(constraints["step"]),
                    step=1,
                    key=f"{unique_key}_step"
                )
            else:
                with col1:
                    param_min = st.number_input(
                        "Min",
                        min_value=float(constraints["min"]),
                        max_value=float(constraints["max"]),
                        value=float(constraints["min"]),
                        step=0.1,
                        format="%.2f",
                        key=f"{unique_key}_min"
                    )
                with col2:
                    param_max = st.number_input(
                        "Max",
                        min_value=float(constraints["min"]),
                        max_value=float(constraints["max"]),
                        value=float(constraints["max"]),
                        step=0.1,
                        format="%.2f",
                        key=f"{unique_key}_max"
                    )
                param_step = st.number_input(
                    "Step",
                    min_value=0.01,
                    max_value=max(0.1, (float(constraints["max"]) - float(constraints["min"])) / 2),
                    value=float(constraints["step"]),
                    step=0.01,
                    format="%.2f",
                    key=f"{unique_key}_step"
                )

            # Calcul nb valeurs
            if param_max > param_min and param_step > 0:
                nb_values = int((param_max - param_min) / param_step) + 1
                st.caption(f"→ {nb_values} valeurs à tester")
            else:
                nb_values = 1
                st.warning("⚠️ Plage invalide")

            return {"min": param_min, "max": param_max, "step": param_step, "count": nb_values}


def create_constrained_slider(
    name: str,
    granularity: float,
    key_prefix: str = ""
) -> Any:
    """
    Wrapper de compatibilité - utilise le nouveau sélecteur en mode single.
    """
    return create_param_range_selector(name, key_prefix, mode="single")


def safe_load_data(
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Charge les données avec gestion d'erreurs complète.

    Returns:
        (dataframe, status_message)
    """
    try:
        df = load_ohlcv(symbol, timeframe, start=start, end=end)

        if df is None or df.empty:
            return None, "Données vides ou fichier non trouvé"

        # Validation du DataFrame
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Colonnes manquantes: {missing}"

        if not isinstance(df.index, pd.DatetimeIndex):
            return None, "L'index n'est pas un DatetimeIndex"

        # Vérifier les données NaN
        nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if nan_pct > 10:
            return None, f"Trop de valeurs NaN ({nan_pct:.1f}%)"

        start_fmt = df.index[0].strftime('%Y-%m-%d')
        end_fmt = df.index[-1].strftime('%Y-%m-%d')
        return df, f"OK: {len(df)} barres ({start_fmt} → {end_fmt})"

    except FileNotFoundError:
        return None, f"Fichier non trouvé: {symbol}_{timeframe}"
    except Exception as e:
        return None, f"Erreur: {str(e)}"


def _data_cache_key(
    symbol: str,
    timeframe: str,
    start_date: Optional[object],
    end_date: Optional[object],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    start_str = str(start_date) if start_date else None
    end_str = str(end_date) if end_date else None
    return (symbol, timeframe, start_str, end_str)


def load_selected_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[object],
    end_date: Optional[object],
) -> Tuple[Optional[pd.DataFrame], str]:
    start_str = str(start_date) if start_date else None
    end_str = str(end_date) if end_date else None
    df, msg = safe_load_data(symbol, timeframe, start_str, end_str)
    if df is not None:
        st.session_state["ohlcv_df"] = df
        st.session_state["ohlcv_cache_key"] = _data_cache_key(
            symbol, timeframe, start_date, end_date
        )
        st.session_state["ohlcv_status_msg"] = msg
    return df, msg


def safe_run_backtest(
    engine: BacktestEngine,
    df: pd.DataFrame,
    strategy: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    run_id: Optional[str] = None,
) -> Tuple[Optional[RunResult], str]:
    """
    Exécute un backtest avec gestion d'erreurs complète.

    Returns:
        (result, status_message)
    """
    # Générer run_id pour corrélation des logs
    run_id = run_id or generate_run_id()
    logger = get_obs_logger("ui.app", run_id=run_id, strategy=strategy, symbol=symbol)

    logger.info("ui_backtest_start params=%s", params)

    try:
        # Passer run_id au moteur pour corrélation
        engine.run_id = run_id
        engine.logger = get_obs_logger("backtest.engine", run_id=run_id)

        result = engine.run(
            df=df,
            strategy=strategy,
            params=params,
            symbol=symbol,
            timeframe=timeframe
        )

        pnl = result.metrics.get("total_pnl", 0)
        sharpe = result.metrics.get("sharpe_ratio", 0)

        logger.info("ui_backtest_end pnl=%.2f sharpe=%.2f", pnl, sharpe)
        return result, f"Terminé | P&L: ${pnl:,.2f} | Sharpe: {sharpe:.2f}"

    except ValueError as e:
        logger.warning("ui_backtest_validation_error error=%s", str(e))
        return None, f"Paramètres invalides: {str(e)}"
    except Exception as e:
        logger.error("ui_backtest_error error=%s", str(e))
        return None, f"Erreur: {str(e)}\n{traceback.format_exc()}"


def _strip_global_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("fees_bps", "slippage_bps", "initial_capital"):
        params.pop(key, None)
    return params


def build_strategy_params_for_comparison(
    strategy_key: str,
    use_preset: bool = True,
) -> Dict[str, Any]:
    try:
        strategy_class = get_strategy(strategy_key)
    except Exception:
        return {}
    if not strategy_class:
        return {}
    strategy_instance = strategy_class()
    params = dict(strategy_instance.default_params)
    if use_preset:
        preset = strategy_instance.get_preset()
        if preset is not None:
            params.update(preset.get_default_values())
    return _strip_global_params(params)


def _aggregate_metric(
    values: List[Any],
    method: str,
    higher_is_better: bool,
) -> float:
    cleaned: List[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        cleaned.append(val)

    if not cleaned:
        return float("nan")

    if method == "median":
        return float(statistics.median(cleaned))
    if method == "worst":
        return float(min(cleaned) if higher_is_better else max(cleaned))
    return float(sum(cleaned) / len(cleaned))


def summarize_comparison_results(
    results: List[Dict[str, Any]],
    aggregate: str,
    primary_metric: str,
    expected_runs: int,
) -> List[Dict[str, Any]]:
    metric_directions = {
        "sharpe_ratio": 1,
        "total_return_pct": 1,
        "win_rate": 1,
        "total_pnl": 1,
        "trades": 1,
        "max_drawdown": -1,
    }
    metrics = [
        "sharpe_ratio",
        "total_return_pct",
        "max_drawdown",
        "win_rate",
        "total_pnl",
        "trades",
    ]
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_strategy.setdefault(item["strategy"], []).append(item)

    summary: List[Dict[str, Any]] = []
    for strategy_key, runs in by_strategy.items():
        row: Dict[str, Any] = {
            "strategy": strategy_key,
            "runs": len(runs),
        }
        if expected_runs > 0:
            row["coverage_pct"] = (len(runs) / expected_runs) * 100
        for metric in metrics:
            values = []
            for run in runs:
                if metric == "trades":
                    values.append(run.get("trades"))
                else:
                    values.append(run.get("metrics", {}).get(metric))
            row[metric] = _aggregate_metric(
                values,
                aggregate,
                metric_directions.get(metric, 1) >= 0,
            )
        summary.append(row)

    direction = metric_directions.get(primary_metric, 1)
    reverse = direction >= 0

    def _sort_key(item: Dict[str, Any]) -> float:
        value = item.get(primary_metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("-inf") if reverse else float("inf")
        return float(value)

    summary.sort(key=_sort_key, reverse=reverse)
    return summary


def build_indicator_overlays(
    strategy_key: str,
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    overlays: Dict[str, Any] = {}
    if df is None or df.empty:
        return overlays

    params = _strip_global_params(dict(params))

    try:
        if strategy_key == "bollinger_atr":
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            entry_z = float(params.get("entry_z", bb_std))
            atr_period = int(params.get("atr_period", 14))
            atr_percentile = float(params.get("atr_percentile", 30))

            bb_upper, bb_mid, bb_lower = calculate_indicator(
                "bollinger",
                df,
                {"period": bb_period, "std_dev": bb_std},
            )
            upper = pd.Series(bb_upper, index=df.index)
            mid = pd.Series(bb_mid, index=df.index)
            lower = pd.Series(bb_lower, index=df.index)
            if bb_std > 0:
                std = (upper - mid) / bb_std
            else:
                std = pd.Series(0.0, index=df.index)
            entry_upper = mid + std * entry_z
            entry_lower = mid - std * entry_z
            overlays["bollinger"] = {
                "upper": upper,
                "lower": lower,
                "mid": mid,
                "entry_upper": entry_upper,
                "entry_lower": entry_lower,
            }

            atr_values = calculate_indicator(
                "atr",
                df,
                {"period": atr_period},
            )
            atr_series = pd.Series(atr_values, index=df.index)
            atr_clean = atr_series.dropna()
            threshold = (
                float(np.nanpercentile(atr_clean, atr_percentile))
                if not atr_clean.empty
                else None
            )
            overlays["atr"] = {"atr": atr_series, "threshold": threshold}

        elif strategy_key == "ema_cross":
            fast_period = int(params.get("fast_period", 12))
            slow_period = int(params.get("slow_period", 26))
            close = df["close"]
            overlays["ema"] = {
                "fast": close.ewm(span=fast_period, adjust=False).mean(),
                "slow": close.ewm(span=slow_period, adjust=False).mean(),
            }

        elif strategy_key == "macd_cross":
            fast_period = int(params.get("fast_period", 12))
            slow_period = int(params.get("slow_period", 26))
            signal_period = int(params.get("signal_period", 9))
            macd_result = calculate_indicator(
                "macd",
                df,
                {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                },
            )
            overlays["macd"] = {
                "macd": pd.Series(macd_result["macd"], index=df.index),
                "signal": pd.Series(macd_result["signal"], index=df.index),
            }

        elif strategy_key == "rsi_reversal":
            rsi_period = int(params.get("rsi_period", 14))
            oversold = float(params.get("oversold_level", 30))
            overbought = float(params.get("overbought_level", 70))
            rsi_values = calculate_indicator(
                "rsi",
                df,
                {"period": rsi_period},
            )
            overlays["rsi"] = {
                "rsi": pd.Series(rsi_values, index=df.index),
                "oversold": oversold,
                "overbought": overbought,
            }

        elif strategy_key == "rsi_trend_filtered":
            rsi_period = int(params.get("rsi_period", 14))
            oversold = float(params.get("oversold_level", 30))
            overbought = float(params.get("overbought_level", 70))
            ema_fast = int(params.get("ema_fast", 20))
            ema_slow = int(params.get("ema_slow", 50))
            rsi_values = calculate_indicator(
                "rsi",
                df,
                {"period": rsi_period},
            )
            close = df["close"]
            overlays["rsi"] = {
                "rsi": pd.Series(rsi_values, index=df.index),
                "oversold": oversold,
                "overbought": overbought,
            }
            overlays["ema"] = {
                "fast": close.ewm(span=ema_fast, adjust=False).mean(),
                "slow": close.ewm(span=ema_slow, adjust=False).mean(),
            }

        elif strategy_key == "ma_crossover":
            fast_period = int(params.get("fast_period", 10))
            slow_period = int(params.get("slow_period", 30))
            close = df["close"]
            overlays["ma"] = {
                "fast": close.rolling(
                    window=fast_period, min_periods=fast_period
                ).mean(),
                "slow": close.rolling(
                    window=slow_period, min_periods=slow_period
                ).mean(),
            }

        elif strategy_key == "ema_stochastic_scalp":
            fast_ema = int(params.get("fast_ema", 50))
            slow_ema = int(params.get("slow_ema", 100))
            stoch_k = int(params.get("stoch_k", 14))
            stoch_d = int(params.get("stoch_d", 3))
            oversold = float(params.get("stoch_oversold", 20))
            overbought = float(params.get("stoch_overbought", 80))
            close = df["close"]
            overlays["ema"] = {
                "fast": close.ewm(span=fast_ema, adjust=False).mean(),
                "slow": close.ewm(span=slow_ema, adjust=False).mean(),
            }
            stoch_values = calculate_indicator(
                "stochastic",
                df,
                {"k_period": stoch_k, "d_period": stoch_d, "smooth_k": 3},
            )
            if isinstance(stoch_values, tuple) and len(stoch_values) >= 2:
                overlays["stochastic"] = {
                    "k": pd.Series(stoch_values[0], index=df.index),
                    "d": pd.Series(stoch_values[1], index=df.index),
                    "oversold": oversold,
                    "overbought": overbought,
                }

        elif strategy_key == "bollinger_dual":
            bb_window = int(params.get("bb_window", 20))
            bb_std = float(params.get("bb_std", 2.0))
            ma_window = int(params.get("ma_window", 10))
            ma_type = str(params.get("ma_type", "sma")).lower()
            upper, middle, lower = calculate_indicator(
                "bollinger",
                df,
                {"period": bb_window, "std_dev": bb_std},
            )
            overlays["bollinger"] = {
                "upper": pd.Series(upper, index=df.index),
                "lower": pd.Series(lower, index=df.index),
                "mid": pd.Series(middle, index=df.index),
            }
            close = df["close"]
            if ma_type == "ema":
                ma_series = close.ewm(span=ma_window, adjust=False).mean()
            else:
                ma_series = close.rolling(
                    window=ma_window, min_periods=ma_window
                ).mean()
            overlays["ma"] = {"center": ma_series}

        elif strategy_key == "atr_channel":
            atr_period = int(params.get("atr_period", 14))
            atr_mult = float(params.get("atr_mult", 2.0))
            close = df["close"]
            ema_center = close.ewm(span=atr_period, adjust=False).mean()
            atr_values = calculate_indicator("atr", df, {"period": atr_period})
            atr_series = pd.Series(atr_values, index=df.index)
            overlays["atr_channel"] = {
                "upper": ema_center + atr_series * atr_mult,
                "lower": ema_center - atr_series * atr_mult,
                "center": ema_center,
            }
            overlays["atr"] = {"atr": atr_series}
    except Exception:
        return {}

    return overlays


# ============================================================================
# VÉRIFICATION BACKEND
# ============================================================================

if not BACKEND_AVAILABLE:
    st.error("❌ Backend non disponible")
    st.code(IMPORT_ERROR)
    st.stop()


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("📈 Backtest Core - Moteur Simplifié")

# Status bar
status_container = st.container()

st.markdown("""
Interface avec validation des paramètres et feedback utilisateur.
Le système de granularité limite le nombre de valeurs testables.
""")

# ============================================================================
# BOUTONS DE CONTRÔLE
# ============================================================================

# Initialiser l'état d'exécution dans session_state
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# Zone des boutons de contrôle
st.markdown("---")
col_btn1, col_btn2, col_spacer = st.columns([2, 2, 6])

with col_btn1:
    run_button = st.button(
        "🚀 Lancer le Backtest",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True,
        key="btn_run_backtest"
    )

with col_btn2:
    stop_button = st.button(
        "⛔ Arrêt d'urgence",
        type="secondary",
        disabled=not st.session_state.is_running,
        use_container_width=True,
        key="btn_stop_backtest"
    )

def _safe_copy_cleanup(logger=None) -> None:
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:
        if logger:
            logger.debug("CuPy import failed (ignored): %s", exc)
        return

    has_pool = hasattr(cp, "get_default_memory_pool") and hasattr(
        cp, "get_default_pinned_memory_pool"
    )
    if not has_pool:
        if logger:
            logger.warning(
                "CuPy cleanup skipped: missing memory pool API. cupy_file=%s",
                getattr(cp, "__file__", None),
            )
        return

    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        if logger:
            logger.debug("CuPy cleanup done: freed default pools.")
    except Exception as exc:
        if logger:
            logger.warning("CuPy cleanup failed (ignored): %s", exc)

# Si arrêt demandé
if stop_button:
    st.session_state.stop_requested = True
    st.session_state.is_running = False

    # Nettoyage RAM/VRAM
    import gc
    gc.collect()

    # Nettoyage CUDA si disponible
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            st.success("✅ VRAM GPU vidée")
    except ImportError:
        pass

    # Nettoyage CuPy si disponible
    import logging
    logger = logging.getLogger(__name__)
    _safe_copy_cleanup(logger)

    st.success("✅ RAM système vidée")
    st.info("💡 Système prêt pour un nouveau test")
    st.session_state.stop_requested = False
    st.rerun()

st.markdown("---")


# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Toggle Debug Mode
with st.sidebar.expander("🔧 Debug", expanded=False):
    debug_enabled = st.checkbox("Mode DEBUG", value=is_debug_enabled(), key="debug_toggle")
    if debug_enabled:
        set_log_level("DEBUG")
        st.caption("🟢 Logs détaillés activés")
    else:
        set_log_level("INFO")

# Granularité désactivée - contrôle par paramètre individuel
granularity = 0.0  # Mode fin par défaut pour max de combinaisons


# --- Section Données ---
st.sidebar.subheader("📊 Données")

# Découverte des données avec gestion d'erreur
data_status = st.sidebar.empty()
try:
    available_tokens, available_timeframes = discover_available_data()
    if not available_tokens:
        available_tokens = ["BTCUSDC", "ETHUSDC"]
        data_status.warning("Aucune donnée trouvée, utilisation des défauts")
    else:
        data_status.success(f"✅ {len(available_tokens)} symboles disponibles")

    if not available_timeframes:
        available_timeframes = ["1h", "4h", "1d"]

except Exception as e:
    available_tokens = ["BTCUSDC", "ETHUSDC"]
    available_timeframes = ["1h", "4h", "1d"]
    data_status.error(f"Erreur scan: {e}")

# Sélection du symbole avec BTCUSDC par défaut
btc_idx = (
    available_tokens.index("BTCUSDC")
    if "BTCUSDC" in available_tokens else 0
)
symbol = st.sidebar.selectbox("Symbole", available_tokens, index=btc_idx)

# Sélection du timeframe avec 30m par défaut
tf_idx = (
    available_timeframes.index("30m")
    if "30m" in available_timeframes else 0
)
timeframe = st.sidebar.selectbox(
    "Timeframe",
    available_timeframes,
    index=tf_idx
)

# Filtre de dates optionnel (activé par défaut sur janvier 2025)
use_date_filter = st.sidebar.checkbox(
    "Filtrer par dates",
    value=True,
    help="Désactivé = utilise toutes les données disponibles"
)
if use_date_filter:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Début", value=pd.Timestamp("2025-01-01"))
    with col2:
        end_date = st.date_input("Fin", value=pd.Timestamp("2025-01-31"))
else:
    start_date = None
    end_date = None

current_data_key = _data_cache_key(symbol, timeframe, start_date, end_date)
if st.session_state.get("ohlcv_cache_key") != current_data_key:
    st.session_state["ohlcv_cache_key"] = current_data_key
    st.session_state["ohlcv_df"] = None
    st.session_state["last_run_result"] = None
    st.session_state["last_winner_params"] = None
    st.session_state["last_winner_metrics"] = None
    st.session_state["last_winner_origin"] = None
    st.session_state["last_winner_meta"] = None

if st.sidebar.button("Charger donnees", key="load_ohlcv_button"):
    df_loaded, msg = load_selected_data(
        symbol, timeframe, start_date, end_date
    )
    if df_loaded is None:
        st.sidebar.error(f"Erreur chargement: {msg}")
    else:
        st.sidebar.success(f"Donnees chargees: {msg}")
else:
    if st.session_state.get("ohlcv_df") is None:
        st.sidebar.info("Donnees non chargees.")
    else:
        cached_msg = st.session_state.get("ohlcv_status_msg", "")
        if cached_msg:
            st.sidebar.caption(f"Cache: {cached_msg}")


# --- Section Stratégie ---
st.sidebar.subheader("🎯 Stratégie")

available_strategies = list_strategies()
strategy_display = {
    "bollinger_atr": "📉 Bollinger + ATR (Mean Reversion)",
    "ema_cross": "📈 EMA Crossover (Trend Following)",
    "macd_cross": "📊 MACD Crossover (Momentum)",
    "rsi_reversal": "🔄 RSI Reversal (Mean Reversion)",
    "atr_channel": "📏 ATR Channel (Breakout)",
    "ma_crossover": "📐 MA Crossover (SMA Trend)",
    "ema_stochastic_scalp": "⚡ EMA + Stochastic (Scalping)",
    "bollinger_dual": "📊 Bollinger Dual (Mean Reversion)",
    "rsi_trend_filtered": "🔄 RSI Trend Filtered (Mean Rev.)",
}

# Types de stratégies pour génération dynamique du tableau
strategy_types = {
    "bollinger_atr": "Mean Rev.",
    "ema_cross": "Trend",
    "macd_cross": "Momentum",
    "rsi_reversal": "Mean Rev.",
    "atr_channel": "Breakout",
    "ma_crossover": "Trend",
    "ema_stochastic_scalp": "Scalping",
    "bollinger_dual": "Mean Rev.",
    "rsi_trend_filtered": "Mean Rev.",
}

strategy_options = {
    strategy_display.get(k, k): k for k in available_strategies
}
strategy_name = st.sidebar.selectbox(
    "Stratégie", list(strategy_options.keys())
)
strategy_key = strategy_options[strategy_name]

# Description de la stratégie
strategy_descriptions = {
    "bollinger_atr": "Achète bas des bandes, vend haut. Filtre ATR.",
    "ema_cross": "Achète sur golden cross EMA, vend sur death cross.",
    "macd_cross": "Achète MACD > Signal, vend MACD < Signal.",
    "rsi_reversal": "Achète RSI<30, vend RSI>70.",
    "atr_channel": "Breakout EMA+ATR: achat haut, vente bas.",
    "ma_crossover": "Croisement SMA rapide/lente.",
    "ema_stochastic_scalp": "Scalping: EMA 50/100 + Stochastic.",
    "bollinger_dual": "Double condition Bollinger + MA crossover.",
    "rsi_trend_filtered": "RSI filtré par tendance EMA.",
}
st.sidebar.caption(strategy_descriptions.get(strategy_key, ""))

# Affichage automatique des indicateurs requis pour la stratégie sélectionnée
strategy_info = None
try:
    strategy_info = get_strategy_info(strategy_key)

    # Afficher les indicateurs requis
    if strategy_info.required_indicators:
        indicators_list = ", ".join([f"**{ind.upper()}**" for ind in strategy_info.required_indicators])
        st.sidebar.info(f"📊 Indicateurs requis: {indicators_list}")
    else:
        st.sidebar.info("📊 Indicateurs: Calculés internement")

    # Afficher les indicateurs calculés internement (info supplémentaire)
    if strategy_info.internal_indicators:
        internal_list = ", ".join([f"{ind.upper()}" for ind in strategy_info.internal_indicators])
        st.sidebar.caption(f"_Calculés: {internal_list}_")

except KeyError:
    # Stratégie pas encore dans le mapping
    st.sidebar.warning(f"⚠️ Indicateurs non définis pour '{strategy_key}'")


# --- Section Indicateurs ---
st.sidebar.subheader("Indicateurs")
strategy_indicator_options = {
    "bollinger_atr": ["bollinger", "atr"],
    "ema_cross": ["ema"],
    "macd_cross": ["macd"],
    "rsi_reversal": ["rsi"],
    "atr_channel": ["atr_channel", "atr"],
    "rsi_trend_filtered": ["rsi", "ema"],
    "ma_crossover": ["ma"],
    "ema_stochastic_scalp": ["ema", "stochastic"],
    "bollinger_dual": ["bollinger", "ma"],
}
available_indicators = strategy_indicator_options.get(strategy_key, [])
active_indicators: List[str] = []

if available_indicators:
    for indicator_name in available_indicators:
        checkbox_key = f"{strategy_key}_indicator_{indicator_name}"
        if st.sidebar.checkbox(
            indicator_name,
            value=True,
            key=checkbox_key,
        ):
            active_indicators.append(indicator_name)
else:
    st.sidebar.caption("Aucun indicateur disponible.")


# --- Section Presets Versionnes ---
st.sidebar.subheader("Versioned presets")

versioned_presets = list_strategy_versions(strategy_key)

# Synchroniser les selectbox après une sauvegarde réussie (avant le rendu des widgets)
if "_sync_preset_version" in st.session_state:
    st.session_state["versioned_preset_version"] = st.session_state.pop("_sync_preset_version")
if "_sync_preset_name" in st.session_state:
    st.session_state["versioned_preset_name"] = st.session_state.pop("_sync_preset_name")

last_saved = st.session_state.pop(
    "versioned_preset_last_saved", None
)
if last_saved:
    st.sidebar.success(f"Preset saved: {last_saved}")

if versioned_presets:
    versions = []
    for preset in versioned_presets:
        meta = preset.metadata or {}
        version = meta.get("version")
        if version and version not in versions:
            versions.append(version)

    default_version = resolve_latest_version(strategy_key)
    if default_version in versions:
        default_index = versions.index(default_version)
    else:
        default_index = 0

    if (
        "versioned_preset_version" in st.session_state
        and st.session_state["versioned_preset_version"] not in versions
    ):
        del st.session_state["versioned_preset_version"]

    selected_version = st.sidebar.selectbox(
        "Preset version",
        versions,
        index=default_index,
        key="versioned_preset_version",
    )

    presets_for_version = [
        p for p in versioned_presets
        if (p.metadata or {}).get("version") == selected_version
    ]
    preset_names = [p.name for p in presets_for_version]

    if (
        "versioned_preset_name" in st.session_state
        and st.session_state["versioned_preset_name"] not in preset_names
    ):
        del st.session_state["versioned_preset_name"]

    selected_preset_name = st.sidebar.selectbox(
        "Preset",
        preset_names,
        key="versioned_preset_name",
    )

    selected_preset = next(
        (p for p in presets_for_version if p.name == selected_preset_name),
        None,
    )

    if selected_preset is not None:
        meta = selected_preset.metadata or {}
        created_at = meta.get("created_at", "")
        if created_at:
            st.sidebar.caption(f"Created: {created_at}")

        indicators = selected_preset.indicators or []
        if indicators:
            st.sidebar.caption(f"Indicators: {', '.join(indicators)}")

        params_values = selected_preset.get_default_values()
        if params_values:
            st.sidebar.json(params_values)

        metrics = meta.get("metrics") or {}
        summary_keys = [
            "sharpe_ratio",
            "total_return_pct",
            "max_drawdown",
            "win_rate",
        ]
        summary = {k: metrics.get(k) for k in summary_keys if k in metrics}
        if summary:
            st.sidebar.json(summary)

    if st.sidebar.button("Load versioned preset", key="load_versioned_preset"):
        try:
            loaded_preset = load_strategy_version(
                strategy_name=strategy_key,
                version=selected_version,
                preset_name=selected_preset_name,
            )
            apply_versioned_preset(loaded_preset, strategy_key)
            st.session_state["loaded_versioned_preset"] = (
                loaded_preset.to_dict()
            )
            st.sidebar.success("Versioned preset loaded")
        except Exception as exc:
            st.sidebar.error(f"Failed to load preset: {exc}")
else:
    st.sidebar.caption("No versioned presets found.")


# --- Section Mode (AVANT les paramètres pour savoir quel mode afficher) ---
st.sidebar.subheader("🔄 Mode d'exécution")

# Initialiser le mode par défaut dans session_state
if "optimization_mode" not in st.session_state:
    st.session_state.optimization_mode = "Backtest Simple"

# Style CSS pour les boutons de mode
st.sidebar.markdown("""
<style>
    .mode-button {
        width: 100%;
        padding: 12px 16px;
        margin: 6px 0;
        border: 2px solid transparent;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
        transition: all 0.3s ease;
    }
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .mode-inactive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        opacity: 0.6;
    }
    .mode-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        opacity: 1;
        border-color: #ffd700;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Créer les boutons de mode (Deep Trace intégré dans mode LLM)
modes = [
    ("Backtest Simple", "📊", "1 combinaison de paramètres"),
    ("Grille de Paramètres", "🔢", "Exploration min/max/step"),
    ("🤖 Optimisation LLM", "🧠", "Agents IA + Deep Trace intégré"),
]

for mode_name, icon, description in modes:
    button_key = f"mode_btn_{mode_name}"
    is_active = st.session_state.optimization_mode == mode_name

    col1, col2 = st.sidebar.columns([1, 10])
    with col1:
        st.write(icon)
    with col2:
        if st.button(
            mode_name,
            key=button_key,
            help=description,
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.optimization_mode = mode_name
            st.rerun()

# Récupérer le mode sélectionné
optimization_mode = st.session_state.optimization_mode

st.sidebar.caption(f"ℹ️ Mode actif: **{optimization_mode}**")

# Définir max_combos avec valeur par défaut
max_combos = 2000000  # Valeur par défaut augmentée
n_workers = 30  # Valeur par défaut augmentée

if optimization_mode == "Grille de Paramètres":
    max_combos = st.sidebar.number_input(
        "Max combinaisons",
        min_value=10,
        max_value=2000000,
        value=2000000,
        step=10000,
        help="Limite pour éviter les temps d'exécution trop longs (10 - 2,000,000)"
    )

    n_workers = st.sidebar.slider(
        "Workers parallèles",
        min_value=1,
        max_value=32,
        value=30,
        help="Nombre de processus parallèles pour l'optimisation (30 recommandé)"
    )

# --- Configuration LLM (si mode LLM sélectionné) ---
llm_config = None
llm_max_iterations = 10
llm_use_walk_forward = True
role_model_config = None  # Configuration multi-modèles par rôle
llm_compare_enabled = False
llm_compare_auto_run = True
llm_compare_strategies: List[str] = []
llm_compare_tokens: List[str] = []
llm_compare_timeframes: List[str] = []
llm_compare_metric = "sharpe_ratio"
llm_compare_aggregate = "median"
llm_compare_max_runs = 25
llm_compare_use_preset = True
llm_compare_generate_report = True
llm_use_multi_agent = False
llm_use_multi_model = False
llm_limit_small_models = False

if optimization_mode == "🤖 Optimisation LLM":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Configuration LLM")

    # === NOUVELLE SECTION: Paramètres d'exécution ===
    st.sidebar.markdown("---")
    st.sidebar.caption("**⚙️ Paramètres d'exécution**")

    max_combos = st.sidebar.number_input(
        "Max combinaisons",
        min_value=10,
        max_value=2000000,
        value=2000000,
        step=10000,
        help="Nombre maximum de backtests que le LLM peut lancer (10 - 2,000,000)",
        key="llm_max_combos"
    )

    n_workers = st.sidebar.slider(
        "Workers parallèles",
        min_value=1,
        max_value=32,
        value=30,
        help="Nombre de backtests exécutés en parallèle (30 recommandé)",
        key="llm_n_workers"
    )

    st.sidebar.caption(
        f"🔧 Parallélisation: jusqu'à {n_workers} backtests simultanés"
    )
    # === FIN NOUVELLE SECTION ===
    st.sidebar.markdown("---")

    if not LLM_AVAILABLE:
        st.sidebar.error("❌ Module LLM non disponible")
        st.sidebar.caption(f"Erreur: {LLM_IMPORT_ERROR}")
    else:
        llm_provider = st.sidebar.selectbox(
            "Provider LLM",
            ["Ollama (Local)", "OpenAI"],
            help="Ollama = gratuit et local | OpenAI = API payante"
        )

        llm_use_multi_agent = st.sidebar.checkbox(
            "Mode multi-agents",
            value=False,
            key="llm_use_multi_agent",
            help="Utiliser Analyst/Strategist/Critic/Validator"
        )

        def _extract_model_params_b(model_name: str) -> Optional[float]:
            match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
            if match:
                return float(match.group(1))
            return None

        def _is_model_under_limit(model_name: str, limit: float) -> bool:
            size = _extract_model_params_b(model_name)
            if size is None:
                return False
            return size < limit

        def _is_model_over_limit(model_name: str, limit: float) -> bool:
            size = _extract_model_params_b(model_name)
            if size is None:
                return False
            return size >= limit

        if "Ollama" in llm_provider:
            # Vérifier la connexion Ollama
            if is_ollama_available():
                st.sidebar.success("✅ Ollama connecté")
            else:
                st.sidebar.warning("⚠️ Ollama non détecté")
                if st.sidebar.button("🚀 Démarrer Ollama"):
                    with st.spinner("Démarrage d'Ollama..."):
                        success, msg = ensure_ollama_running()
                        if success:
                            st.sidebar.success(msg)
                            st.rerun()
                        else:
                            st.sidebar.error(msg)

            # === CONFIGURATION MULTI-MODELES PAR ROLE ===
            llm_use_multi_model = False
            if llm_use_multi_agent:
                llm_use_multi_model = st.sidebar.checkbox(
                    "Multi-modeles par role",
                    value=False,
                    key="llm_use_multi_model",
                    help="Assigner differents modeles a chaque role d'agent"
                )

            if llm_use_multi_model:
                # Charger les modeles disponibles
                available_models_list = list_available_models()
                available_model_names = [m.name for m in available_models_list]

                llm_limit_small_models = st.sidebar.checkbox(
                    "Limiter selection aleatoire a <20B",
                    value=True,
                    key="llm_limit_small_models",
                    help="Filtre la liste par taille et exclut deepseek-r1:70b"
                )
                llm_limit_large_models = st.sidebar.checkbox(
                    "Limiter selection aleatoire a >=20B",
                    value=False,
                    key="llm_limit_large_models",
                    help="Filtre la liste par taille (>=20B uniquement)"
                )

                effective_small_filter = llm_limit_small_models
                effective_large_filter = llm_limit_large_models
                if effective_small_filter and effective_large_filter:
                    st.sidebar.warning(
                        "Filtres <20B et >=20B actifs: >=20B prioritaire."
                    )
                    effective_small_filter = False

                excluded_models = set()
                if not effective_large_filter:
                    excluded_models = {"deepseek-r1:70b"}
                if excluded_models:
                    available_model_names = [
                        m for m in available_model_names if m not in excluded_models
                    ]

                if effective_small_filter:
                    filtered = [
                        m for m in available_model_names if _is_model_under_limit(m, 20)
                    ]
                    if filtered:
                        available_model_names = filtered
                    else:
                        st.sidebar.warning(
                            "Aucun modele <20B detecte, filtre desactive."
                        )

                if effective_large_filter:
                    filtered = [
                        m for m in available_model_names if _is_model_over_limit(m, 20)
                    ]
                    if filtered:
                        available_model_names = filtered
                    else:
                        available_model_names = []
                        st.sidebar.warning(
                            "Aucun modele >=20B detecte."
                        )
                if effective_large_filter and not available_model_names:
                    st.sidebar.error(
                        "Selection >=20B activee mais aucun modele compatible."
                    )

                # Categoriser pour l'affichage
                light_models = [m.name for m in available_models_list if m.category == ModelCategory.LIGHT]
                medium_models = [m.name for m in available_models_list if m.category == ModelCategory.MEDIUM]
                heavy_models = [m.name for m in available_models_list if m.category == ModelCategory.HEAVY]

                st.sidebar.markdown("---")
                st.sidebar.caption("**Modeles par role d'agent**")
                st.sidebar.caption("Rapide | Moyen | Lent")

                # Initialiser la config
                role_model_config = get_global_model_config()

                # Helper pour afficher le badge de categorie
                def model_with_badge(name: str) -> str:
                    info = KNOWN_MODELS.get(name)
                    if info:
                        if info.category == ModelCategory.LIGHT:
                            return f"[L] {name}"
                        elif info.category == ModelCategory.MEDIUM:
                            return f"[M] {name}"
                        else:
                            return f"[H] {name}"
                    return name

                # Convertir pour l'affichage
                model_options_display = [model_with_badge(m) for m in available_model_names]
                name_to_display = {n: model_with_badge(n) for n in available_model_names}
                display_to_name = {v: k for k, v in name_to_display.items()}

                # ANALYST - Modèles rapides recommandés
                st.sidebar.markdown("**Analyst** (analyse rapide)")
                analyst_defaults = [name_to_display.get(m, m) for m in role_model_config.analyst.models if m in available_model_names]
                analyst_default_options = analyst_defaults[:3] if analyst_defaults else model_options_display[:2]
                if not model_options_display:
                    analyst_default_options = []
                analyst_selection = st.sidebar.multiselect(
                    "Modeles Analyst",
                    model_options_display,
                    default=analyst_default_options,
                    key="analyst_models",
                    help="Modeles rapides recommandes pour l'analyse"
                )

                # STRATEGIST - Modèles moyens
                st.sidebar.markdown("**Strategist** (propositions)")
                strategist_defaults = [name_to_display.get(m, m) for m in role_model_config.strategist.models if m in available_model_names]
                strategist_default_options = strategist_defaults[:3] if strategist_defaults else model_options_display[:2]
                if not model_options_display:
                    strategist_default_options = []
                strategist_selection = st.sidebar.multiselect(
                    "Modeles Strategist",
                    model_options_display,
                    default=strategist_default_options,
                    key="strategist_models",
                    help="Modeles moyens pour la creativite"
                )

                # CRITIC - Modèles puissants
                st.sidebar.markdown("**Critic** (evaluation critique)")
                critic_defaults = [name_to_display.get(m, m) for m in role_model_config.critic.models if m in available_model_names]
                critic_default_options = critic_defaults[:3] if critic_defaults else model_options_display[:2]
                if not model_options_display:
                    critic_default_options = []
                critic_selection = st.sidebar.multiselect(
                    "Modeles Critic",
                    model_options_display,
                    default=critic_default_options,
                    key="critic_models",
                    help="Modeles puissants pour la reflexion"
                )

                # VALIDATOR - Modèles puissants
                st.sidebar.markdown("**Validator** (decision finale)")
                validator_defaults = [name_to_display.get(m, m) for m in role_model_config.validator.models if m in available_model_names]
                validator_default_options = validator_defaults[:3] if validator_defaults else model_options_display[:2]
                if not model_options_display:
                    validator_default_options = []
                validator_selection = st.sidebar.multiselect(
                    "Modeles Validator",
                    model_options_display,
                    default=validator_default_options,
                    key="validator_models",
                    help="Modeles puissants pour decisions finales"
                )

                # Option: Autoriser les modeles lourds apres N iterations
                st.sidebar.markdown("---")
                st.sidebar.caption("Modeles lourds")
                heavy_after_iter = st.sidebar.number_input(
                    "Autoriser apres iteration N",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="Les modeles lourds ne seront utilises qu'apres cette iteration"
                )

                # Mettre a jour la configuration (filtrer strictement la selection)
                def _normalize_selection(selection: List[str]) -> List[str]:
                    names = [display_to_name.get(m, m) for m in selection]
                    return [n for n in names if n in available_model_names]

                role_model_config.analyst.models = _normalize_selection(analyst_selection)
                role_model_config.strategist.models = _normalize_selection(strategist_selection)
                role_model_config.critic.models = _normalize_selection(critic_selection)
                role_model_config.validator.models = _normalize_selection(validator_selection)

                # Appliquer le seuil des modeles lourds
                for assignment in [role_model_config.analyst, role_model_config.strategist,
                                   role_model_config.critic, role_model_config.validator]:
                    assignment.allow_heavy_after_iteration = heavy_after_iter

                # Sauvegarder globalement
                set_global_model_config(role_model_config)

                # Info sur la selection aleatoire
                st.sidebar.info(
                    "Si plusieurs modeles sont selectionnes, "
                    "un sera choisi aleatoirement a chaque appel."
                )

                # Modele par defaut (premier de Analyst)
                if role_model_config.analyst.models:
                    llm_model = role_model_config.analyst.models[0]
                elif available_model_names:
                    llm_model = available_model_names[0]
                elif effective_large_filter:
                    llm_model = None
                else:
                    llm_model = "deepseek-r1:8b"

            else:
                # === MODE SIMPLE: UN SEUL MODÈLE ===
                # Sélecteur de modèles avec liste dynamique
                available_models = get_available_models_for_ui(
                    preferred_order=RECOMMENDED_FOR_STRATEGY
                )

                llm_model = st.sidebar.selectbox(
                    "Modèle Ollama",
                    available_models,
                    help="Modèles installés localement via Ollama"
                )

                # Afficher les infos du modèle sélectionné
                if llm_model:
                    model_info = get_model_info(llm_model)
                    size = model_info["size_gb"]
                    desc = model_info["description"]
                    st.sidebar.caption(f"📦 ~{size} GB | {desc}")

            ollama_host = st.sidebar.text_input(
                "URL Ollama",
                value="http://localhost:11434",
                help="Adresse du serveur Ollama"
            )
            if llm_model:
                llm_config = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=llm_model,
                    ollama_host=ollama_host
                )
            else:
                llm_config = None
        else:
            openai_key = st.sidebar.text_input(
                "Clé API OpenAI",
                type="password",
                help="Votre clé API OpenAI"
            )
            llm_model = st.sidebar.selectbox(
                "Modèle OpenAI",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                help="gpt-4o-mini recommandé pour coût/performance"
            )
            if openai_key:
                llm_config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model=llm_model,
                    api_key=openai_key
                )
            else:
                st.sidebar.warning("⚠️ Clé API requise")

        st.sidebar.markdown("---")
        st.sidebar.caption("**Options d'optimisation**")

        llm_max_iterations = st.sidebar.slider(
            "Max itérations",
            min_value=3,
            max_value=50,
            value=10,
            help="Nombre max de cycles d'amélioration"
        )

        llm_use_walk_forward = st.sidebar.checkbox(
            "Walk-Forward Validation",
            value=True,
            help="Anti-overfitting: valide sur données hors-échantillon"
        )

        llm_unload_during_backtest = st.sidebar.checkbox(
            "🎮 Décharger LLM du GPU",
            value=False,
            help=(
                "Libère la VRAM GPU pendant les backtests pour améliorer les performances. "
                "Recommandé si vous utilisez CuPy/GPU pour les indicateurs. "
                "Désactivé par défaut (compatibilité CPU-only)."
            )
        )

        st.sidebar.markdown("---")
        with st.sidebar.expander("Comparaison multi-strategies", expanded=False):
            llm_compare_enabled = st.checkbox(
                "Comparer strategies (multi-tokens/timeframes)",
                value=False,
                key="llm_compare_enabled",
            )
            if llm_compare_enabled:
                llm_compare_auto_run = st.checkbox(
                    "Execution automatique",
                    value=True,
                    key="llm_compare_auto_run",
                    help="Lance la comparaison avant l'optimisation LLM",
                )
                compare_strategy_labels = st.multiselect(
                    "Strategies a comparer",
                    list(strategy_options.keys()),
                    default=[strategy_name],
                    key="llm_compare_strategy_labels",
                )
                llm_compare_strategies = [
                    strategy_options[label]
                    for label in compare_strategy_labels
                    if label in strategy_options
                ]

                llm_compare_tokens = st.multiselect(
                    "Tokens",
                    available_tokens,
                    default=[symbol],
                    key="llm_compare_tokens",
                )
                llm_compare_timeframes = st.multiselect(
                    "Timeframes",
                    available_timeframes,
                    default=[timeframe],
                    key="llm_compare_timeframes",
                )

                llm_compare_metric = st.selectbox(
                    "Metrica principale",
                    [
                        "sharpe_ratio",
                        "total_return_pct",
                        "max_drawdown",
                        "win_rate",
                    ],
                    index=0,
                    key="llm_compare_metric",
                )
                llm_compare_aggregate = st.selectbox(
                    "Agregation",
                    ["median", "mean", "worst"],
                    index=0,
                    key="llm_compare_aggregate",
                )
                llm_compare_max_runs = st.number_input(
                    "Max runs comparaison",
                    min_value=1,
                    max_value=500,
                    value=25,
                    step=1,
                    key="llm_compare_max_runs",
                )
                llm_compare_use_preset = st.checkbox(
                    "Utiliser presets si disponibles",
                    value=True,
                    key="llm_compare_use_preset",
                )
                llm_compare_generate_report = st.checkbox(
                    "Generer justification LLM",
                    value=True,
                    key="llm_compare_generate_report",
                )

                if (
                    llm_compare_strategies
                    and llm_compare_tokens
                    and llm_compare_timeframes
                ):
                    total_runs = (
                        len(llm_compare_strategies)
                        * len(llm_compare_tokens)
                        * len(llm_compare_timeframes)
                    )
                    st.caption(
                        f"Estime: {total_runs} runs (cap {llm_compare_max_runs})."
                    )

                if not llm_compare_auto_run:
                    if "llm_compare_run_now" not in st.session_state:
                        st.session_state["llm_compare_run_now"] = False
                    if st.button("Lancer comparaison", key="llm_compare_run_button"):
                        st.session_state["llm_compare_run_now"] = True
            else:
                if "llm_compare_run_now" in st.session_state:
                    st.session_state["llm_compare_run_now"] = False

        if llm_use_multi_agent:
            st.sidebar.caption(
                f"Agents: Analyst/Strategist/Critic/Validator | "
                f"Max iterations: {llm_max_iterations}"
            )
        else:
            st.sidebar.caption(
                f"Agent autonome | Max iterations: {llm_max_iterations}"
            )


# --- Section Paramètres ---
st.sidebar.subheader("🔧 Paramètres")

# Déterminer le mode de sélection des paramètres
param_mode = "range" if optimization_mode == "Grille de Paramètres" else "single"

params = {}
param_ranges = {}  # Pour le mode grille: stocke min/max/step
param_specs: Dict[str, Any] = {}
strategy_class = get_strategy(strategy_key)
strategy_instance = None

if strategy_class:
    temp_strategy = strategy_class()
    strategy_instance = temp_strategy
    param_specs = temp_strategy.parameter_specs or {}

    if param_specs:
        validation_errors = []

        for param_name, spec in param_specs.items():
            if param_name == "leverage":
                continue  # Géré séparément

            if param_mode == "single":
                value = create_param_range_selector(
                    param_name, strategy_key, mode="single", spec=spec
                )
                if value is not None:
                    params[param_name] = value

                    # Validation en temps réel
                    is_valid, error = validate_param(param_name, value)
                    if not is_valid:
                        validation_errors.append(error)
            else:
                # Mode grille: sélection de plages
                range_data = create_param_range_selector(
                    param_name, strategy_key, mode="range", spec=spec
                )
                if range_data is not None:
                    param_ranges[param_name] = range_data
                    # Valeur par défaut pour le backtest
                    if spec is not None:
                        params[param_name] = spec.default
                    else:
                        params[param_name] = PARAM_CONSTRAINTS[param_name]["default"]

        # Afficher erreurs de validation
        if validation_errors:
            for err in validation_errors:
                st.sidebar.error(err)

        # Estimation combinaisons via compute_search_space_stats() (unifié)
        if param_mode == "range" and param_ranges:
            st.sidebar.markdown("---")
            # Utiliser la fonction unifiée
            stats = compute_search_space_stats(param_ranges, max_combinations=max_combos)

            if stats.is_continuous:
                st.sidebar.info("ℹ️ Espace continu détecté")
            elif stats.has_overflow:
                st.sidebar.warning(f"⚠️ {stats.total_combinations:,} combinaisons (limite: {max_combos:,})")
                st.sidebar.caption("Réduisez les plages ou augmentez le step")
            else:
                st.sidebar.success(f"✅ {stats.total_combinations:,} combinaisons à tester")

            # Afficher détail par paramètre (expandeur)
            with st.sidebar.expander("📊 Détail par paramètre"):
                for pname, pcount in stats.per_param_counts.items():
                    st.caption(f"• {pname}: {pcount} valeurs")
        else:
            st.sidebar.caption("📊 Mode simple: 1 combinaison")
else:
    st.sidebar.error(f"Stratégie '{strategy_key}' non trouvée")


# --- Section Trading ---
st.sidebar.subheader("💰 Trading")

leverage = create_param_range_selector("leverage", "trading", mode="single")
params["leverage"] = leverage

initial_capital = st.sidebar.number_input(
    "Capital Initial ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Capital de départ (1,000 - 1,000,000)"
)

st.markdown("---")
st.subheader("Schema indicateurs & parametres")
if strategy_instance is None:
    st.info("Selectionnez une strategie pour afficher le schema.")
else:
    diagram_params = {
        **strategy_instance.default_params,
        **params,
    }
    render_strategy_param_diagram(
        strategy_key,
        diagram_params,
        key=f"diagram_{strategy_key}",
    )

st.markdown("---")
st.subheader("Apercu OHLCV + indicateurs")
preview_df = st.session_state.get("ohlcv_df")
if preview_df is None:
    st.info("Chargez les donnees pour afficher l'apercu.")
else:
    preview_overlays = build_indicator_overlays(
        strategy_key, preview_df, params
    )
    render_ohlcv_with_trades_and_indicators(
        df=preview_df,
        trades_df=pd.DataFrame(),
        overlays=preview_overlays,
        active_indicators=active_indicators,
        title="OHLCV + indicateurs (apercu)",
        key="ohlcv_indicator_preview",
        height=650,
    )


# ============================================================================
# ZONE PRINCIPALE - EXÉCUTION ET RÉSULTATS
# ============================================================================

# Modes normaux (Simple/Grid/LLM)
result = st.session_state.get("last_run_result")
winner_params = st.session_state.get("last_winner_params")
winner_metrics = st.session_state.get("last_winner_metrics")
winner_origin = st.session_state.get("last_winner_origin")
winner_meta = st.session_state.get("last_winner_meta")

if run_button:
    # Activer le flag d'exécution
    st.session_state.is_running = True
    st.session_state.stop_requested = False
    winner_params = None
    winner_metrics = None
    winner_origin = None
    winner_meta = None

    # Validation globale des paramètres
    is_valid, errors = validate_all_params(params)

    if not is_valid:
        with status_container:
            show_status("error", "Paramètres invalides")
            for err in errors:
                st.error(f"  • {err}")
        st.session_state.is_running = False
        st.stop()

    # Étape 1: Chargement des données
    with st.spinner("📥 Chargement des données..."):
        df = st.session_state.get("ohlcv_df")
        data_msg = st.session_state.get("ohlcv_status_msg", "")

        if df is None:
            df, data_msg = load_selected_data(
                symbol, timeframe, start_date, end_date
            )

        if df is None:
            with status_container:
                show_status("error", f"Échec chargement: {data_msg}")
                st.info(
                    "💡 Vérifiez les fichiers dans "
                    "`D:\\ThreadX_big\\data\\crypto\\processed\\parquet\\`"
                )
            st.session_state.is_running = False
            st.stop()

        if df is not None:
            with status_container:
                show_status("success", f"Données chargées: {data_msg}")

    # Créer le moteur
    engine = BacktestEngine(initial_capital=initial_capital)

    if optimization_mode == "Backtest Simple":
        # ===== MODE SIMPLE =====

        with st.spinner("⚙️ Exécution du backtest..."):
            result, result_msg = safe_run_backtest(
                engine, df, strategy_key, params, symbol, timeframe
            )

        if result is None:
            with status_container:
                show_status("error", f"Échec backtest: {result_msg}")
            st.session_state.is_running = False
            st.stop()

        with status_container:
            show_status("success", f"Backtest terminé: {result_msg}")
        winner_params = result.meta.get("params", params)
        winner_metrics = result.metrics
        winner_origin = "backtest"
        winner_meta = result.meta
        st.session_state["last_run_result"] = result
        st.session_state["last_winner_params"] = winner_params
        st.session_state["last_winner_metrics"] = winner_metrics
        st.session_state["last_winner_origin"] = winner_origin
        st.session_state["last_winner_meta"] = winner_meta

    elif optimization_mode == "Grille de Paramètres":
        # ===== MODE GRILLE =====

        with st.spinner("📊 Génération de la grille..."):
            try:
                # Générer la grille à partir des param_ranges définis par l'utilisateur
                param_grid = []
                param_names = list(param_ranges.keys())

                if param_names:
                    # Créer les listes de valeurs pour chaque paramètre
                    param_values_lists = []
                    for pname in param_names:
                        r = param_ranges[pname]
                        pmin, pmax, pstep = r["min"], r["max"], r["step"]

                        # Générer les valeurs
                        if isinstance(pmin, int) and isinstance(pstep, int):
                            values = list(range(int(pmin), int(pmax) + 1, int(pstep)))
                        else:
                            values = list(np.arange(float(pmin), float(pmax) + float(pstep)/2, float(pstep)))
                            values = [round(v, 2) for v in values if v <= pmax]

                        param_values_lists.append(values)

                    # Produit cartésien
                    for combo in product(*param_values_lists):
                        param_dict = dict(zip(param_names, combo))
                        # Ajouter leverage
                        param_dict["leverage"] = params.get("leverage", 1)
                        param_grid.append(param_dict)
                else:
                    # Aucun paramètre en mode range, utiliser les valeurs par défaut
                    param_grid = [params.copy()]

                if len(param_grid) > max_combos:
                    st.warning(
                        f"⚠️ Grille limitée: {len(param_grid):,} → {max_combos:,}"
                    )
                    param_grid = param_grid[:max_combos]

                show_status(
                    "info", f"Grille: {len(param_grid):,} combinaisons"
                )

            except Exception as e:
                show_status("error", f"Échec génération grille: {e}")
                st.session_state.is_running = False
                st.stop()

        # Exécution de la grille (parallèle si workers > 1)
        results_list = []
        param_combos_map = {}  # Pour retrouver le dict original

        # Créer le moniteur de progression
        monitor = ProgressMonitor(total_runs=len(param_grid))
        monitor_placeholder = st.empty()

        # Affichage initial du moniteur
        st.markdown("### 📊 Progression en temps réel")
        render_progress_monitor(monitor, monitor_placeholder)

        # Fonction pour un seul backtest (pour parallélisation)
        def run_single_backtest(param_combo):
            """Exécute un seul backtest et retourne le résultat."""
            try:
                result_i, msg_i = safe_run_backtest(
                    engine, df, strategy_key, param_combo, symbol, timeframe
                )

                # Convertir np.float64 en float natif
                params_native = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in param_combo.items()
                }
                params_str = str(params_native)

                if result_i:
                    return {
                        "params": params_str,
                        "params_dict": param_combo,
                        "total_pnl": result_i.metrics["total_pnl"],
                        "sharpe": result_i.metrics["sharpe_ratio"],
                        "max_dd": result_i.metrics["max_drawdown"],
                        "win_rate": result_i.metrics["win_rate"],
                        "trades": result_i.metrics["total_trades"],
                        "profit_factor": result_i.metrics["profit_factor"]
                    }
                else:
                    return {
                        "params": params_str,
                        "params_dict": param_combo,
                        "error": msg_i
                    }
            except Exception as e:
                params_str = str(param_combo)
                return {
                    "params": params_str,
                    "params_dict": param_combo,
                    "error": str(e)
                }

        # Exécution parallèle ou séquentielle selon n_workers
        if n_workers > 1 and len(param_grid) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Soumettre tous les jobs
                future_to_params = {
                    executor.submit(run_single_backtest, combo): combo
                    for combo in param_grid
                }

                completed = 0
                last_render_time = time.perf_counter()

                for future in as_completed(future_to_params):
                    completed += 1
                    monitor.runs_completed = completed

                    # Throttling : ne mettre à jour l'affichage que toutes les 0.5s
                    current_time = time.perf_counter()
                    if current_time - last_render_time >= 0.5:
                        render_progress_monitor(monitor, monitor_placeholder)
                        last_render_time = current_time
                        time.sleep(0.01)  # Laisser Streamlit rafraîchir

                    result = future.result()
                    params_str = result.get("params", "")
                    param_combos_map[params_str] = result.get("params_dict", {})

                    # Retirer params_dict du résultat final
                    result_clean = {k: v for k, v in result.items() if k != "params_dict"}
                    results_list.append(result_clean)

                # Dernier render pour afficher 100%
                render_progress_monitor(monitor, monitor_placeholder)
        else:
            # Exécution séquentielle
            last_render_time = time.perf_counter()

            for i, param_combo in enumerate(param_grid):
                monitor.runs_completed = i + 1

                # Throttling : ne mettre à jour l'affichage que toutes les 0.5s
                current_time = time.perf_counter()
                if current_time - last_render_time >= 0.5:
                    render_progress_monitor(monitor, monitor_placeholder)
                    last_render_time = current_time
                    time.sleep(0.01)  # Laisser Streamlit rafraîchir

                result = run_single_backtest(param_combo)
                params_str = result.get("params", "")
                param_combos_map[params_str] = result.get("params_dict", {})

                result_clean = {k: v for k, v in result.items() if k != "params_dict"}
                results_list.append(result_clean)

            # Dernier render pour afficher 100%
            render_progress_monitor(monitor, monitor_placeholder)

        # Nettoyer l'affichage du moniteur
        monitor_placeholder.empty()

        # Afficher résultats grille
        with status_container:
            show_status(
                "success", f"Optimisation: {len(results_list)} tests"
            )

        results_df = pd.DataFrame(results_list)

        # 🔍 DEBUG: Logging pour investiguer le bug "37.5 trades"
        if "trades" in results_df.columns:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("=" * 80)
            logger.info("🔍 DEBUG GRID SEARCH - Analyse de la colonne 'trades'")
            logger.info(f"   Type: {results_df['trades'].dtype}")
            logger.info(f"   Shape: {results_df['trades'].shape}")
            logger.info(f"   Premières valeurs: {results_df['trades'].head(10).tolist()}")
            logger.info(f"   Stats: min={results_df['trades'].min()}, max={results_df['trades'].max()}, mean={results_df['trades'].mean():.2f}")

            # Vérifier si il y a des floats
            trades_values = results_df['trades'].values
            fractional = [x for x in trades_values if isinstance(x, float) and not x.is_integer()]
            if fractional:
                logger.warning(f"   ⚠️  {len(fractional)} valeurs fractionnaires détectées: {fractional[:5]}")
            else:
                logger.info("   ✅ Toutes les valeurs sont des entiers")
            logger.info("=" * 80)

        error_column = results_df.get("error")
        if error_column is not None:
            valid_results = results_df[error_column.isna()]
        else:
            valid_results = results_df

        if not valid_results.empty:
            valid_results = valid_results.sort_values(
                "sharpe", ascending=False
            )

            st.subheader("🏆 Top 10 Combinaisons")

            # 🔍 Afficher les infos de debug dans l'UI (optionnel)
            with st.expander("🔍 Debug Info - Types de données"):
                st.text(f"Nombre de résultats: {len(valid_results)}")
                st.text("Types des colonnes:")
                st.text(str(valid_results.dtypes))
                if "trades" in valid_results.columns:
                    st.text("\nStatistiques 'trades':")
                    st.text(f"  Type: {valid_results['trades'].dtype}")
                    st.text(f"  Min: {valid_results['trades'].min()}")
                    st.text(f"  Max: {valid_results['trades'].max()}")
                    st.text(f"  Mean: {valid_results['trades'].mean():.2f}")

            st.dataframe(valid_results.head(10), width="stretch")

            # Relancer avec le meilleur
            best = valid_results.iloc[0]
            st.info(f"🥇 Meilleure: {best['params']}")

            # Récupérer le dict original depuis le mapping
            best_params = param_combos_map.get(best['params'], {})
            result, _ = safe_run_backtest(
                engine, df, strategy_key, best_params, symbol, timeframe
            )
            if result is not None:
                winner_params = best_params
                winner_metrics = result.metrics
                winner_origin = "grid"
                winner_meta = result.meta
                st.session_state["last_run_result"] = result
                st.session_state["last_winner_params"] = winner_params
                st.session_state["last_winner_metrics"] = winner_metrics
                st.session_state["last_winner_origin"] = winner_origin
                st.session_state["last_winner_meta"] = winner_meta
        else:
            show_status("error", "Aucun résultat valide")
            st.session_state.is_running = False
            st.stop()

    elif optimization_mode == "🤖 Optimisation LLM":
        # ===== MODE OPTIMISATION LLM =====

        if not LLM_AVAILABLE:
            show_status("error", "Module agents LLM non disponible")
            st.code(LLM_IMPORT_ERROR)
            st.session_state.is_running = False
            st.stop()

        if llm_config is None:
            show_status("error", "Configuration LLM incomplète")
            st.info("Configurez le provider LLM dans la sidebar")
            st.session_state.is_running = False
            st.stop()

        # Créer le logger d'orchestration
        session_id = generate_session_id()
        orchestration_logger = OrchestrationLogger(session_id=session_id)

        # Récupérer les bornes des paramètres pour la stratégie
        try:
            param_bounds = get_strategy_param_bounds(strategy_key)
            if not param_bounds:
                # Fallback: créer des bornes depuis PARAM_CONSTRAINTS
                param_bounds = {}
                for pname in params.keys():
                    if pname in PARAM_CONSTRAINTS:
                        c = PARAM_CONSTRAINTS[pname]
                        param_bounds[pname] = (c["min"], c["max"])
        except Exception as e:
            show_status("warning", f"Bornes par défaut utilisées: {e}")
            param_bounds = {}
            for pname in params.keys():
                if pname in PARAM_CONSTRAINTS:
                    c = PARAM_CONSTRAINTS[pname]
                    param_bounds[pname] = (c["min"], c["max"])

        # Calculer l'estimation d'espace discret si step disponible
        try:
            full_param_space = get_strategy_param_space(strategy_key, include_step=True)
            llm_space_stats = compute_search_space_stats(full_param_space)
        except Exception:
            llm_space_stats = None

        max_iterations = min(llm_max_iterations, max_combos)

        comparison_summary: List[Dict[str, Any]] = []
        should_run_comparison = llm_compare_enabled and (
            llm_compare_auto_run
            or st.session_state.get("llm_compare_run_now", False)
        )
        if should_run_comparison:
            st.subheader("Comparaison multi-strategies")
            if not llm_compare_strategies:
                st.warning("Aucune strategie selectionnee pour la comparaison.")
            elif not llm_compare_tokens or not llm_compare_timeframes:
                st.warning("Selectionnez au moins un token et un timeframe.")
            else:
                start_str = str(start_date) if start_date else None
                end_str = str(end_date) if end_date else None
                progress_bar = st.progress(0)
                comparison_results: List[Dict[str, Any]] = []
                comparison_errors: List[str] = []
                data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

                for token in llm_compare_tokens:
                    for tf in llm_compare_timeframes:
                        df_cmp, msg = safe_load_data(token, tf, start_str, end_str)
                        if df_cmp is None:
                            comparison_errors.append(f"{token}/{tf}: {msg}")
                        else:
                            data_cache[(token, tf)] = df_cmp

                valid_pairs = list(data_cache.keys())
                total_runs = len(valid_pairs) * len(llm_compare_strategies)
                total_runs = max(0, min(total_runs, llm_compare_max_runs))
                run_index = 0

                with st.spinner("Comparaison en cours..."):
                    for strategy_name_cmp in llm_compare_strategies:
                        params_cmp = build_strategy_params_for_comparison(
                            strategy_name_cmp,
                            use_preset=llm_compare_use_preset,
                        )
                        for token, tf in valid_pairs:
                            if run_index >= total_runs:
                                break
                            df_cmp = data_cache[(token, tf)]
                            result_cmp, status = safe_run_backtest(
                                engine,
                                df_cmp,
                                strategy_name_cmp,
                                params_cmp,
                                token,
                                tf,
                            )
                            if result_cmp is None:
                                comparison_errors.append(
                                    f"{strategy_name_cmp} {token}/{tf}: {status}"
                                )
                            else:
                                comparison_results.append(
                                    {
                                        "strategy": strategy_name_cmp,
                                        "symbol": token,
                                        "timeframe": tf,
                                        "metrics": result_cmp.metrics,
                                        "trades": len(result_cmp.trades),
                                    }
                                )
                            run_index += 1
                            if total_runs > 0:
                                progress_bar.progress(run_index / total_runs)
                        if run_index >= total_runs:
                            break

                if comparison_errors:
                    st.warning(
                        "Comparaison: "
                        + "; ".join(comparison_errors[:8])
                        + (" ..." if len(comparison_errors) > 8 else "")
                    )

                if comparison_results:
                    comparison_summary = summarize_comparison_results(
                        comparison_results,
                        aggregate=llm_compare_aggregate,
                        primary_metric=llm_compare_metric,
                        expected_runs=len(valid_pairs),
                    )
                    st.caption(
                        f"Runs effectues: {len(comparison_results)} / {total_runs}"
                    )
                    st.dataframe(
                        pd.DataFrame(comparison_summary),
                        width="stretch",
                    )

                    chart_rows = []
                    for row in comparison_summary:
                        chart_rows.append(
                            {
                                "name": row["strategy"],
                                "metrics": {
                                    llm_compare_metric: row.get(llm_compare_metric)
                                },
                            }
                        )
                    render_comparison_chart(
                        chart_rows,
                        metric=llm_compare_metric,
                        title="Comparaison agregree",
                        key="llm_strategy_comparison",
                    )

                    if llm_compare_generate_report:
                        try:
                            llm_client = create_llm_client(llm_config)
                            if not llm_client.is_available():
                                st.warning(
                                    "LLM indisponible pour la justification."
                                )
                            else:
                                summary_lines = [
                                    "strategy | runs | sharpe | return_pct | max_drawdown | win_rate"
                                ]
                                for row in comparison_summary:
                                    summary_lines.append(
                                        f"{row.get('strategy')} | "
                                        f"{row.get('runs')} | "
                                        f"{row.get('sharpe_ratio', float('nan')):.2f} | "
                                        f"{row.get('total_return_pct', float('nan')):.2f} | "
                                        f"{row.get('max_drawdown', float('nan')):.2f} | "
                                        f"{row.get('win_rate', float('nan')):.1f}"
                                    )

                                system_prompt = (
                                    "You are a senior quantitative strategist. "
                                    "Compare strategy robustness across assets and timeframes."
                                )
                                user_message = (
                                    "Comparison scope:\n"
                                    f"- tokens: {', '.join(llm_compare_tokens)}\n"
                                    f"- timeframes: {', '.join(llm_compare_timeframes)}\n"
                                    f"- aggregation: {llm_compare_aggregate}\n"
                                    f"- primary metric: {llm_compare_metric}\n\n"
                                    "Summary table (metrics are percent where applicable):\n"
                                    + "\n".join(summary_lines)
                                    + "\n\n"
                                    "Provide:\n"
                                    "1) Ranking with short justification.\n"
                                    "2) Notes on robustness and risk.\n"
                                    "3) Which strategies deserve further optimization."
                                )

                                response = llm_client.simple_chat(
                                    user_message=user_message,
                                    system_prompt=system_prompt,
                                    temperature=0.3,
                                )
                                st.markdown("**Justification LLM**")
                                st.write(response.content)
                        except Exception as exc:
                            st.warning(
                                f"Justification LLM indisponible: {exc}"
                            )
                st.session_state["llm_compare_run_now"] = False

        # Interface d'optimisation LLM
        st.subheader("🤖 Optimisation par Agents LLM")

        col_info, col_timeline = st.columns([1, 2])

        with col_info:
            st.markdown(f"""
            **Stratégie:** `{strategy_key}`
            **Paramètres initiaux:** `{params}`
            **Max itérations:** {llm_max_iterations}
            **Walk-Forward:** {'✅' if llm_use_walk_forward else '❌'}
            """)

            st.markdown("**Bornes des paramètres:**")
            for pname, (pmin, pmax) in param_bounds.items():
                st.caption(f"• {pname}: [{pmin}, {pmax}]")

            # Afficher estimation d'espace si disponible
            if llm_space_stats:
                st.markdown("---")
                if llm_space_stats.is_continuous:
                    st.info("ℹ️ **Espace continu** : exploration adaptative par LLM")
                else:
                    st.caption(f"📊 Espace discret estimé: ~{llm_space_stats.total_combinations:,} combinaisons")
                    st.caption("_(Le LLM explore de façon intelligente sans énumérer)_")

        # Zone de timeline des agents
        timeline_container = col_timeline.empty()

        strategist = None
        executor = None
        orchestrator = None

        # Vérifier les doublons de runs
        run_tracker = get_global_tracker()
        # Construire un identifiant de données basé sur les métriques du DataFrame
        data_identifier = f"df_{len(df)}rows_{df.index[0]}_{df.index[-1]}" if len(df) > 0 else "empty_df"
        run_signature = RunSignature(
            strategy_name=strategy_key,
            data_path=data_identifier,
            initial_params=params,
            llm_model=llm_model,
            mode="multi_agents" if llm_use_multi_agent else "autonomous",
            session_id=session_id,
        )

        if run_tracker.is_duplicate(run_signature):
            st.warning(
                "⚠️ **Configuration déjà testée !**\n\n"
                "Cette combinaison stratégie/données/paramètres a déjà été optimisée précédemment. "
                "Relancer le même run pourrait être redondant."
            )

            # Afficher les runs similaires
            similar_runs = run_tracker.find_similar(run_signature)
            if similar_runs:
                with st.expander("📋 Runs similaires détectés", expanded=True):
                    for i, prev_run in enumerate(similar_runs[-3:], 1):
                        st.caption(
                            f"{i}. {prev_run.timestamp[:19]} - "
                            f"Mode: {prev_run.mode} - "
                            f"Modèle: {prev_run.llm_model or 'N/A'}"
                        )

            # Demander confirmation
            if not st.checkbox("⚠️ Je confirme vouloir relancer malgré tout", key="confirm_duplicate"):
                st.stop()

        # Enregistrer ce run
        run_tracker.register(run_signature)

        # Créer l'optimiseur
        with st.spinner("🔌 Connexion au LLM..."):
            try:
                if llm_use_multi_agent:
                    # Live viewer multi-agents (même expérience que mono LLM)
                    live_events_placeholder = st.empty()
                    live_viewer = LiveOrchestrationViewer(container_key="live_orch_viewer_multi")

                    def on_orchestration_event(entry):
                        live_viewer.add_event(entry)
                        live_viewer.render(live_events_placeholder, show_header=True)

                    orchestration_logger.set_on_event_callback(on_orchestration_event)

                    orchestrator = create_orchestrator_with_backtest(
                        llm_config=llm_config,
                        strategy_name=strategy_key,
                        data=df,
                        initial_params=params,
                        role_model_config=role_model_config,
                        use_walk_forward=llm_use_walk_forward,
                        orchestration_logger=orchestration_logger,
                        session_id=session_id,
                        n_workers=n_workers,
                        max_iterations=max_iterations,
                        initial_capital=initial_capital,
                        config=engine.config,
                    )
                    show_status(
                        "success", "Connexion LLM établie (mode multi-agents)"
                    )
                else:
                    strategist, executor = create_optimizer_from_engine(
                        llm_config=llm_config,
                        strategy_name=strategy_key,
                        data=df,
                        initial_capital=initial_capital,
                        use_walk_forward=llm_use_walk_forward,
                        verbose=True,
                        unload_llm_during_backtest=llm_unload_during_backtest,
                        orchestration_logger=orchestration_logger,
                    )
                    show_status("success", "Connexion LLM établie")
            except Exception as e:
                show_status("error", f"Echec connexion LLM: {e}")
                st.code(traceback.format_exc())
                st.session_state.is_running = False
                st.stop()

        if llm_use_multi_agent:
            st.markdown("---")
            st.markdown("### Progression multi-agents")
            st.caption(
                f"Limite: {max_combos:,} backtests max, "
                f"{n_workers} workers, {max_iterations} iterations max"
            )

            if orchestrator is None:
                show_status("error", "Orchestrator non initialise")
                st.session_state.is_running = False
                st.stop()

            try:
                with st.spinner("Optimisation multi-agents en cours..."):
                    orchestrator_result = orchestrator.run()

                # Forcer une persistance complète en fin de run
                try:
                    orchestration_logger.save_to_jsonl()
                except Exception:
                    pass

                if orchestrator_result.errors:
                    st.warning(f"Orchestration errors: {len(orchestrator_result.errors)}")
                if orchestrator_result.warnings:
                    st.warning(f"Orchestration warnings: {len(orchestrator_result.warnings)}")

                if orchestrator_result.success:
                    st.success("Optimisation multi-agents terminee")
                else:
                    st.warning(
                        f"Optimisation multi-agents terminee (decision: {orchestrator_result.decision})"
                    )

                if orchestrator_result.final_params:
                    st.subheader("Resultat multi-agents")
                    st.json(orchestrator_result.final_params)
                else:
                    st.warning("Aucun parametre final retourne")

                if orchestrator_result.final_metrics:
                    metrics = orchestrator_result.final_metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Sharpe", f"{metrics.sharpe_ratio:.3f}")
                    with col_b:
                        st.metric("Return", f"{metrics.total_return:.2%}")
                    with col_c:
                        st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")

                if orchestrator_result.iteration_history:
                    st.markdown("---")
                    st.dataframe(
                        pd.DataFrame(orchestrator_result.iteration_history),
                        width="stretch",
                    )

                best_params = orchestrator_result.final_params or {}
                if best_params:
                    result, _ = safe_run_backtest(
                        engine, df, strategy_key, best_params, symbol, timeframe
                    )
                    if result is not None:
                        winner_params = best_params
                        winner_metrics = result.metrics
                        winner_origin = "llm"
                        winner_meta = result.meta
                        st.session_state["last_run_result"] = result
                        st.session_state["last_winner_params"] = winner_params
                        st.session_state["last_winner_metrics"] = winner_metrics
                        st.session_state["last_winner_origin"] = winner_origin
                        st.session_state["last_winner_meta"] = winner_meta
            except Exception as e:
                show_status("error", f"Erreur optimisation multi-agents: {e}")
                st.code(traceback.format_exc())
                st.session_state.is_running = False
                st.stop()
        else:
            # Exécuter l'optimisation
            st.markdown("---")
            st.markdown("### 📊 Progression de l'optimisation LLM")

            # Créer les placeholders pour affichage live
            live_status = st.status("🚀 Démarrage de l'optimisation...", expanded=True)
            live_events_placeholder = st.empty()
            orchestration_placeholder = st.empty()

            # Créer le moniteur de progression (LLM: on suit les itérations)
            max_iterations = min(llm_max_iterations, max_combos)

            # Créer le viewer live pour les événements d'orchestration
            live_viewer = LiveOrchestrationViewer(container_key="live_orch_viewer")

            # Connecter le callback au logger d'orchestration
            def on_orchestration_event(entry):
                """Callback appelé à chaque événement - met à jour l'UI."""
                live_viewer.add_event(entry)
                # Mettre à jour le placeholder avec les derniers événements
                live_viewer.render(live_events_placeholder, show_header=True)

            orchestration_logger.set_on_event_callback(on_orchestration_event)

            # Info configuration
            st.caption(f"🔧 Limite: {max_combos:,} backtests max, {n_workers} workers, {max_iterations} itérations max")

            try:
                # Utiliser st.status pour un affichage en temps réel
                with live_status:
                    st.write("🤖 **Agent LLM actif** - Optimisation autonome")
                    st.write(f"📊 Stratégie: `{strategy_key}` | Modèle: `{llm_model}`")

                    # Lancer l'optimisation autonome avec limite
                    session = strategist.optimize(
                        executor=executor,
                        initial_params=params,
                        param_bounds=param_bounds,
                        max_iterations=max_iterations,
                        min_sharpe=-5.0,  # Assouplir contraintes
                        max_drawdown=0.50,
                    )

                    # Mise à jour du status final
                    live_status.update(
                        label=f"✅ Optimisation terminée en {session.current_iteration} itérations",
                        state="complete",
                        expanded=False
                    )

                st.success(f"✅ Optimisation terminée en {session.current_iteration} itérations")

                # Afficher l'historique des itérations dans un expander
                with st.expander("📝 Historique des itérations", expanded=True):
                    for i, exp in enumerate(session.all_results):
                        icon = "🟢" if exp.sharpe_ratio > 0 else "🔴"
                        col_it1, col_it2, col_it3 = st.columns([2, 1, 1])
                        with col_it1:
                            st.markdown(f"**Itération {i+1}** {icon}")
                            st.caption(f"Params: `{exp.request.parameters}`")
                        with col_it2:
                            st.metric("Sharpe", f"{exp.sharpe_ratio:.3f}")
                        with col_it3:
                            st.metric("Return", f"{exp.total_return:.2%}")

                # Sauvegarder et afficher les logs d'orchestration
                try:
                    orchestration_logger.save_to_jsonl()
                except Exception:
                    pass

                # Afficher le viewer complet des logs d'orchestration
                with orchestration_placeholder:
                    st.markdown("---")

                    # Onglets: Vue simple + Deep Trace avancé
                    tab_simple, tab_deep = st.tabs([
                        "📋 Logs d'orchestration",
                        "🔍 Deep Trace (avancé)"
                    ])

                    with tab_simple:
                        render_full_orchestration_viewer(
                            orchestration_logger=orchestration_logger,
                            max_entries=50
                        )

                    with tab_deep:
                        # Afficher le Deep Trace Viewer complet
                        if LLM_AVAILABLE:
                            render_deep_trace_viewer(logger=orchestration_logger)
                        else:
                            st.warning("Module LLM non disponible pour Deep Trace avancé")

                # Résultats finaux
                st.markdown("---")
                st.subheader("🏆 Résultat de l'optimisation LLM")

                col_best, col_improve = st.columns(2)

                with col_best:
                    st.markdown("**Meilleurs paramètres trouvés:**")
                    st.json(session.best_result.request.parameters)

                    st.metric(
                        "Meilleur Sharpe",
                        f"{session.best_result.sharpe_ratio:.3f}"
                    )
                    st.metric(
                        "Return",
                        f"{session.best_result.total_return:.2%}"
                    )

                with col_improve:
                    # Calculer l'amélioration
                    if session.all_results:
                        initial_sharpe = session.all_results[0].sharpe_ratio
                        best_sharpe = session.best_result.sharpe_ratio
                        improvement = ((best_sharpe - initial_sharpe) / abs(initial_sharpe) * 100) if initial_sharpe != 0 else 0

                        st.metric(
                            "Amélioration Sharpe",
                            f"{improvement:+.1f}%",
                            delta=f"{best_sharpe - initial_sharpe:+.3f}"
                        )
                        st.metric("Itérations utilisées", session.current_iteration)

                        if session.final_reasoning:
                            st.info(f"🛑 Arrêt: {session.final_reasoning}")

                # Relancer le backtest avec les meilleurs paramètres
                best_params = session.best_result.request.parameters
                result, _ = safe_run_backtest(
                    engine, df, strategy_key, best_params, symbol, timeframe
                )
                if result is not None:
                    winner_params = best_params
                    winner_metrics = result.metrics
                    winner_origin = "llm"
                    winner_meta = result.meta
                    st.session_state["last_run_result"] = result
                    st.session_state["last_winner_params"] = winner_params
                    st.session_state["last_winner_metrics"] = winner_metrics
                    st.session_state["last_winner_origin"] = winner_origin
                    st.session_state["last_winner_meta"] = winner_meta

            except Exception as e:
                live_status.update(label=f"❌ Erreur: {e}", state="error")
                show_status("error", f"Erreur optimisation LLM: {e}")
                st.code(traceback.format_exc())
                st.session_state.is_running = False
                st.stop()

    else:
        # Mode non reconnu
        show_status("error", f"Mode non reconnu: {optimization_mode}")
        st.session_state.is_running = False
        st.stop()

st.session_state.is_running = False

result = st.session_state.get("last_run_result")
winner_params = st.session_state.get("last_winner_params")
winner_metrics = st.session_state.get("last_winner_metrics")
winner_origin = st.session_state.get("last_winner_origin")
winner_meta = st.session_state.get("last_winner_meta")

# ============================================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================================

if result is not None:
    st.header("📊 Résultats du Backtest")

    # --- Métriques principales ---
    col1, col2, col3, col4, col5 = st.columns(5)

    if result is not None:
        with col1:
            pnl = result.metrics['total_pnl']
            pnl_color: str = "normal" if pnl >= 0 else "inverse"
            ret_pct = result.metrics['total_return_pct']
            st.metric(
                "P&L Total",
                f"${pnl:,.2f}",
                delta=f"{ret_pct:.1f}%",
                delta_color=pnl_color  # type: ignore[arg-type]
            )

        with col2:
            sharpe = result.metrics['sharpe_ratio']
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with col3:
            max_dd = result.metrics['max_drawdown']
            st.metric("Max Drawdown", f"{max_dd:.1f}%")

        with col4:
            trades = result.metrics['total_trades']
            win_rate = result.metrics['win_rate']
            st.metric("Trades", f"{trades}", delta=f"{win_rate:.0f}% wins")

        with col5:
            best_pnl, best_run_id = _BEST_PNL_TRACKER.get_best()
            if best_pnl is None:
                st.metric("Best PnL seen", "n/a")
            else:
                st.metric("Best PnL seen", f"${best_pnl:,.2f}")
                if best_run_id:
                    st.caption(f"run {best_run_id}")

    if result is not None and winner_params is not None:
        st.subheader("Versioned preset")
        col_save_a, col_save_b = st.columns(2)

        with col_save_a:
            default_version = resolve_latest_version(strategy_key)
            preset_version = st.text_input(
                "Preset version",
                value=default_version,
                key="winner_preset_version",
            )
            preset_name = st.text_input(
                "Preset name",
                value="winner",
                key="winner_preset_name",
            )

        with col_save_b:
            description_default = (
                f"{strategy_key} winner {symbol}/{timeframe}"
            )
            preset_description = st.text_input(
                "Description",
                value=description_default,
                key="winner_preset_description",
            )

        if st.button("Save winner preset", key="save_winner_preset"):
            extra_meta = {}
            if winner_meta:
                for key in [
                    "symbol",
                    "timeframe",
                    "period_start",
                    "period_end",
                ]:
                    if key in winner_meta:
                        extra_meta[key] = winner_meta[key]

            origin_run_id = None
            if winner_meta and "run_id" in winner_meta:
                origin_run_id = winner_meta["run_id"]

            try:
                saved = save_versioned_preset(
                    strategy_name=strategy_key,
                    version=preset_version,
                    preset_name=preset_name,
                    params_values=winner_params,
                    indicators=strategy_info.required_indicators
                    if strategy_info is not None
                    else None,
                    description=preset_description,
                    metrics=winner_metrics,
                    origin=winner_origin,
                    origin_run_id=origin_run_id,
                    extra_metadata=extra_meta,
                )
                # Marquer pour synchronisation au prochain cycle (avant rendu des widgets)
                st.session_state["_sync_preset_version"] = preset_version
                st.session_state["_sync_preset_name"] = saved.name
                st.session_state["versioned_preset_last_saved"] = saved.name
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    # --- Graphique OHLCV avec Trades ---
    st.subheader("📈 Prix et Trades")

    if result is not None:
        chart_df = st.session_state.get("ohlcv_df")
        if chart_df is None:
            st.info("Donnees non chargees. Cliquez sur 'Charger donnees'.")
        else:
            chart_params = result.meta.get("params", params)
            indicator_overlays = build_indicator_overlays(
                strategy_key, chart_df, chart_params
            )

            if indicator_overlays:
                render_ohlcv_with_trades_and_indicators(
                    df=chart_df,
                    trades_df=result.trades,
                    overlays=indicator_overlays,
                    active_indicators=active_indicators,
                    title="📊 OHLCV + Indicateurs + Entrees/Sorties",
                    key="ohlcv_trades_indicators_main",
                    height=700,
                )
            elif not result.trades.empty:
                # Afficher le graphique de bougies avec les points d'entrée/sortie
                render_ohlcv_with_trades(
                    df=chart_df,
                    trades_df=result.trades,
                    title="📊 Graphique OHLCV avec Points d'Entree/Sortie",
                    key="ohlcv_trades_main",
                    height=600,
                )
            else:
                st.info(
                    "Aucun trade execute, affichage du graphique de prix uniquement"
                )
                render_ohlcv_with_trades(
                    df=chart_df,
                    trades_df=pd.DataFrame(),  # DataFrame vide
                    title="📊 Graphique OHLCV",
                    key="ohlcv_main_notrades",
                    height=600,
                )

    # --- Métriques détaillées ---
    st.subheader("📈 Métriques Détaillées")

    if result is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💰 Rendement**")
            st.text(f"P&L Total: ${result.metrics['total_pnl']:,.2f}")
            st.text(f"Rendement: {result.metrics['total_return_pct']:.2f}%")
            st.text(f"Ann. Return: {result.metrics['annualized_return']:.2f}%")
            st.text(f"Volatilité: {result.metrics['volatility_annual']:.2f}%")

        with col2:
            st.markdown("**📊 Risque**")
            st.text(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
            st.text(f"Sortino: {result.metrics['sortino_ratio']:.2f}")
            st.text(f"Calmar: {result.metrics['calmar_ratio']:.2f}")
            st.text(f"Max DD: {result.metrics['max_drawdown']:.2f}%")

        with col3:
            st.markdown("**🎯 Trading**")
            st.text(f"Trades: {result.metrics['total_trades']}")
            st.text(f"Win Rate: {result.metrics['win_rate']:.1f}%")
            st.text(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
            st.text(f"Expectancy: ${result.metrics['expectancy']:.2f}")

    # --- Liste des Trades ---
    if result is not None and not result.trades.empty:
        st.subheader("📋 Historique des Trades")

        trades_display = result.trades.copy()

        # Formater les colonnes
        if "entry_ts" in trades_display.columns:
            trades_display["entry_ts"] = pd.to_datetime(
                trades_display["entry_ts"]
            ).dt.strftime("%Y-%m-%d %H:%M")
        if "exit_ts" in trades_display.columns:
            trades_display["exit_ts"] = pd.to_datetime(
                trades_display["exit_ts"]
            ).dt.strftime("%Y-%m-%d %H:%M")
        if "pnl" in trades_display.columns:
            trades_display["pnl"] = trades_display["pnl"].apply(
                lambda x: f"${x:,.2f}"
            )
        if "price_entry" in trades_display.columns:
            trades_display["price_entry"] = trades_display[
                "price_entry"
            ].apply(lambda x: f"${x:,.2f}")
        if "price_exit" in trades_display.columns:
            trades_display["price_exit"] = trades_display["price_exit"].apply(
                lambda x: f"${x:,.2f}"
            )

        # Sélectionner colonnes à afficher
        cols_to_show = [
            "entry_ts", "exit_ts", "side", "price_entry",
            "price_exit", "pnl", "return_pct", "exit_reason"
        ]
        display_cols = [
            c for c in cols_to_show if c in trades_display.columns
        ]

        st.dataframe(trades_display[display_cols], width="stretch")

        # Stats trades
        total_trades = len(result.trades)
        winners = (result.trades['pnl'] > 0).sum()
        losers = (result.trades['pnl'] < 0).sum()
        st.caption(
            f"Total: {total_trades} | Gagnants: {winners} | Perdants: {losers}"
        )
    elif result is not None:
        st.info("Aucun trade exécuté pendant cette période")

else:
    # ============================================================================
    # ÉCRAN D'ACCUEIL
    # ============================================================================

    st.info("👆 Configurez dans la sidebar puis cliquez sur **🚀 Lancer le Backtest**")

    # Onglets d'information
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Stratégies", "📊 Optimisation", "📁 Données", "❓ FAQ"]
    )

    with tab1:
        # Génération dynamique du tableau des stratégies
        strategies_table = generate_strategies_table()
        st.markdown(strategies_table)

        st.markdown("""
        ### Indicateurs Intégrés
        - Bollinger Bands, ATR, RSI, EMA, SMA, MACD, ADX
        - Ichimoku, PSAR, Stochastic RSI, Vortex, etc.
        """)

    with tab2:
        st.markdown("""
        ### Système d'Optimisation

        **Mode Grille** *(par défaut)* : Test de multiples combinaisons.
        - Définissez Min/Max/Step pour chaque paramètre
        - Le système calcule toutes les combinaisons
        - Limite configurable (jusqu'à 1,000,000)

        **Mode Simple** : Test d'une seule combinaison de paramètres.

        **Mode LLM** 🤖 : Optimisation intelligente par agents IA.
        - 4 agents spécialisés (Analyst, Strategist, Critic, Validator)
        - Boucle d'amélioration itérative automatique
        - Walk-Forward anti-overfitting intégré
        - Supporte Ollama (local/gratuit) ou OpenAI

        | Mode | Combinaisons | Intelligence | Coût |
        |------|--------------|--------------|------|
        | Simple | 1 | ❌ | Gratuit |
        | Grille | Jusqu'à 1M | ❌ | Gratuit |
        | LLM | ~10-50 ciblées | ✅ | Variable |

        ⚠️ Mode LLM nécessite Ollama installé localement ou une clé OpenAI.
        """)

    with tab3:
        st.markdown(f"""
        ### Format des Données

        Les données OHLCV doivent être au format Parquet ou CSV:
        - `SYMBOL_TIMEFRAME.parquet` (ex: `BTCUSDT_1h.parquet`)

        **Symboles détectés**: {len(available_tokens)}
        **Timeframes**: {', '.join(available_timeframes)}
        """)

    with tab4:
        st.markdown("""
        ### Questions Fréquentes

        **Q: Comment tester plus de combinaisons?**
        R: En mode Grille, définissez Min/Max/Step pour chaque paramètre.
        Augmentez la limite de combinaisons si nécessaire.

        **Q: Que signifie le Sharpe Ratio?**
        R: Rendement ajusté au risque. > 1 = bon, > 2 = excellent.

        **Q: Pourquoi le mode Grille est lent?**
        R: Il teste toutes les combinaisons. Augmentez le Step ou réduisez la plage.

        **Q: Comment éviter l'overfitting?**
        R: Utilisez le Walk-Forward Validation (activé par défaut en mode LLM).

        **Q: Comment fonctionne le mode LLM?**
        R: 4 agents IA travaillent ensemble:
        1. **Analyst** analyse les métriques actuelles
        2. **Strategist** propose de nouveaux paramètres
        3. **Critic** détecte l'overfitting potentiel
        4. **Validator** décide: approuver, rejeter ou itérer

        **Q: Ollama vs OpenAI?**
        R: Ollama est gratuit et local (installer depuis ollama.ai).
        OpenAI est plus puissant mais payant (~0.01$/requête).
        """)


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Backtest Core v2.1**")
llm_status = "✅ LLM" if LLM_AVAILABLE else "❌ LLM"
st.sidebar.caption(f"Architecture découplée UI/Moteur | {llm_status}")
