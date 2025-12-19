"""
Backtest Core - Streamlit Application v2
========================================

Interface utilisateur robuste avec:
- Validation des paramètres avec contraintes
- Feedback utilisateur clair (success/error/warning)
- Gestion d'erreurs complète
- Visualisation améliorée des résultats

Lancer avec: streamlit run ui/app.py
"""

import ast
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports du moteur (backend)
IMPORT_ERROR = ""  # Initialisé avant le try/except
try:
    from backtest.engine import BacktestEngine, RunResult
    from backtest.performance import drawdown_series
    from data.loader import discover_available_data, load_ohlcv
    from strategies.base import get_strategy, list_strategies
    from strategies.indicators_mapping import (
        get_required_indicators,
        get_all_indicators,
        get_strategy_info,
    )
    from utils.parameters import (
        calculate_combinations,
        compute_search_space_stats,
        generate_param_grid,
        parameter_values,
        SearchSpaceStats,
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Imports agents LLM (optionnels)
LLM_AVAILABLE = False
LLM_IMPORT_ERROR = ""
try:
    from agents.integration import (
        create_optimizer_from_engine,
        run_backtest_for_agent,
        get_strategy_param_bounds,
        get_strategy_param_space,
    )
    from agents.llm_client import LLMConfig, LLMProvider, create_llm_client
    from agents.autonomous_strategist import AutonomousStrategist, OptimizationSession
    from agents.backtest_executor import BacktestExecutor
    from agents.ollama_manager import (
        ensure_ollama_running,
        is_ollama_available,
        list_ollama_models,
    )
    from agents.model_config import (
        RoleModelConfig,
        ModelCategory,
        ModelInfo,
        list_available_models,
        get_models_by_category,
        get_global_model_config,
        set_global_model_config,
        KNOWN_MODELS,
    )
    from agents.orchestration_logger import OrchestrationLogger, generate_session_id
    from ui.components.agent_timeline import (
        AgentActivityTimeline,
        AgentActivity,
        AgentType,
        ActivityType,
        render_agent_timeline,
        render_mini_timeline,
    )
    from ui.components.monitor import render_mini_monitor
    from ui.components.model_selector import (
        get_available_models_for_ui,
        RECOMMENDED_FOR_STRATEGY,
        get_model_info,
    )
    from ui.orchestration_viewer import (
        render_full_orchestration_viewer,
        render_orchestration_logs,
        render_orchestration_summary_table,
    )
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_IMPORT_ERROR = str(e)

# Import observabilité (toujours disponible)
from utils.observability import (
    init_logging,
    get_obs_logger,
    generate_run_id,
    set_log_level,
    is_debug_enabled,
    build_diagnostic_summary,
    PerfCounters,
)

# Import composants UI (toujours disponibles)
from ui.components.charts import (
    render_equity_and_drawdown,
    render_ohlcv_with_trades,
    render_equity_curve,
)
from ui.components.thinking_viewer import (
    ThinkingStreamViewer,
    render_thinking_stream,
)

# Initialiser le logging au démarrage de l'UI
init_logging()


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
    "atr_period": {
        "min": 2, "max": 100, "step": 1, "default": 14,
        "description": "Période ATR (2-100)"
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


def create_param_range_selector(
    name: str,
    key_prefix: str = "",
    mode: str = "single"  # "single" ou "range"
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
    if name not in PARAM_CONSTRAINTS:
        st.sidebar.warning(f"Paramètre {name} sans contraintes définies")
        return None

    constraints = PARAM_CONSTRAINTS[name]
    unique_key = f"{key_prefix}_{name}"
    is_int = constraints["step"] == 1
    
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
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except ImportError:
        pass

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

# Créer les boutons de mode
modes = [
    ("Backtest Simple", "📊", "1 combinaison de paramètres"),
    ("Grille de Paramètres", "🔢", "Exploration min/max/step"),
    ("🤖 Optimisation LLM", "🧠", "Agents IA autonomes")
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

            # === CONFIGURATION MULTI-MODÈLES PAR RÔLE ===
            use_multi_model = st.sidebar.checkbox(
                "🎭 Multi-modèles par rôle",
                value=False,
                help="Assigner différents modèles à chaque rôle d'agent"
            )
            
            if use_multi_model:
                # Charger les modèles disponibles
                available_models_list = list_available_models()
                available_model_names = [m.name for m in available_models_list]
                
                # Catégoriser pour l'affichage
                light_models = [m.name for m in available_models_list if m.category == ModelCategory.LIGHT]
                medium_models = [m.name for m in available_models_list if m.category == ModelCategory.MEDIUM]
                heavy_models = [m.name for m in available_models_list if m.category == ModelCategory.HEAVY]
                
                st.sidebar.markdown("---")
                st.sidebar.caption("**🎭 Modèles par rôle d'agent**")
                st.sidebar.caption("_🟢 Rapide | 🟡 Moyen | 🔴 Lent_")
                
                # Initialiser la config
                role_model_config = get_global_model_config()
                
                # Helper pour afficher le badge de catégorie
                def model_with_badge(name: str) -> str:
                    info = KNOWN_MODELS.get(name)
                    if info:
                        if info.category == ModelCategory.LIGHT:
                            return f"🟢 {name}"
                        elif info.category == ModelCategory.MEDIUM:
                            return f"🟡 {name}"
                        else:
                            return f"🔴 {name}"
                    return name
                
                # Convertir pour l'affichage
                model_options_display = [model_with_badge(m) for m in available_model_names]
                name_to_display = {n: model_with_badge(n) for n in available_model_names}
                display_to_name = {v: k for k, v in name_to_display.items()}
                
                # ANALYST - Modèles rapides recommandés
                st.sidebar.markdown("**📊 Analyst** _(analyse rapide)_")
                analyst_defaults = [name_to_display.get(m, m) for m in role_model_config.analyst.models if m in available_model_names]
                analyst_selection = st.sidebar.multiselect(
                    "Modèles Analyst",
                    model_options_display,
                    default=analyst_defaults[:3] if analyst_defaults else model_options_display[:2],
                    key="analyst_models",
                    help="Modèles rapides (🟢) recommandés pour l'analyse"
                )
                
                # STRATEGIST - Modèles moyens
                st.sidebar.markdown("**💡 Strategist** _(propositions)_")
                strategist_defaults = [name_to_display.get(m, m) for m in role_model_config.strategist.models if m in available_model_names]
                strategist_selection = st.sidebar.multiselect(
                    "Modèles Strategist",
                    model_options_display,
                    default=strategist_defaults[:3] if strategist_defaults else model_options_display[:2],
                    key="strategist_models",
                    help="Modèles moyens (🟡) pour la créativité"
                )
                
                # CRITIC - Modèles puissants
                st.sidebar.markdown("**🔍 Critic** _(évaluation critique)_")
                critic_defaults = [name_to_display.get(m, m) for m in role_model_config.critic.models if m in available_model_names]
                critic_selection = st.sidebar.multiselect(
                    "Modèles Critic",
                    model_options_display,
                    default=critic_defaults[:3] if critic_defaults else model_options_display[:2],
                    key="critic_models",
                    help="Modèles puissants (🟡/🔴) pour la réflexion"
                )
                
                # VALIDATOR - Modèles puissants (mais pas 70B en premier)
                st.sidebar.markdown("**✅ Validator** _(décision finale)_")
                validator_defaults = [name_to_display.get(m, m) for m in role_model_config.validator.models if m in available_model_names]
                validator_selection = st.sidebar.multiselect(
                    "Modèles Validator",
                    model_options_display,
                    default=validator_defaults[:3] if validator_defaults else model_options_display[:2],
                    key="validator_models",
                    help="Modèles puissants pour décisions finales"
                )
                
                # Option: Autoriser les 70B après N itérations
                st.sidebar.markdown("---")
                st.sidebar.caption("**⚙️ Modèles lourds (70B+)**")
                heavy_after_iter = st.sidebar.number_input(
                    "Autoriser après itération N",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="Les modèles 70B+ ne seront utilisés qu'après cette itération"
                )
                
                # Mettre à jour la configuration
                role_model_config.analyst.models = [display_to_name.get(m, m) for m in analyst_selection]
                role_model_config.strategist.models = [display_to_name.get(m, m) for m in strategist_selection]
                role_model_config.critic.models = [display_to_name.get(m, m) for m in critic_selection]
                role_model_config.validator.models = [display_to_name.get(m, m) for m in validator_selection]
                
                # Appliquer le seuil des modèles lourds
                for assignment in [role_model_config.analyst, role_model_config.strategist, 
                                   role_model_config.critic, role_model_config.validator]:
                    assignment.allow_heavy_after_iteration = heavy_after_iter
                
                # Sauvegarder globalement
                set_global_model_config(role_model_config)
                
                # Info sur la sélection aléatoire
                st.sidebar.info(
                    "💡 Si plusieurs modèles sont sélectionnés, "
                    "un sera choisi **aléatoirement** à chaque appel."
                )
                
                # Modèle par défaut (premier de Analyst)
                llm_model = role_model_config.analyst.models[0] if role_model_config.analyst.models else "deepseek-r1:8b"
            
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
            llm_config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=llm_model,
                ollama_host=ollama_host
            )
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
        
        st.sidebar.caption(
            f"🤖 L'agent va analyser, proposer, critiquer "
            f"et valider jusqu'à {llm_max_iterations} itérations"
        )


# --- Section Paramètres ---
st.sidebar.subheader("🔧 Paramètres")

# Déterminer le mode de sélection des paramètres
param_mode = "range" if optimization_mode == "Grille de Paramètres" else "single"

params = {}
param_ranges = {}  # Pour le mode grille: stocke min/max/step
param_specs: Dict[str, Any] = {}
strategy_class = get_strategy(strategy_key)

if strategy_class:
    temp_strategy = strategy_class()
    param_specs = temp_strategy.parameter_specs or {}

    if param_specs:
        validation_errors = []

        for param_name, spec in param_specs.items():
            if param_name == "leverage":
                continue  # Géré séparément

            if param_mode == "single":
                value = create_param_range_selector(
                    param_name, strategy_key, mode="single"
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
                    param_name, strategy_key, mode="range"
                )
                if range_data is not None:
                    param_ranges[param_name] = range_data
                    # Valeur par défaut pour le backtest
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
            st.sidebar.caption(f"📊 Mode simple: 1 combinaison")
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


# ============================================================================
# ZONE PRINCIPALE - EXÉCUTION ET RÉSULTATS
# ============================================================================

if run_button:
    # Activer le flag d'exécution
    st.session_state.is_running = True
    st.session_state.stop_requested = False

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
        start_str = str(start_date) if start_date else None
        end_str = str(end_date) if end_date else None

        df, data_msg = safe_load_data(symbol, timeframe, start_str, end_str)

        if df is None:
            with status_container:
                show_status("error", f"Échec chargement: {data_msg}")
                st.info(
                    "💡 Vérifiez les fichiers dans "
                    "`D:\\ThreadX_big\\data\\crypto\\processed\\parquet\\`"
                )
            st.session_state.is_running = False
            st.stop()

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

    elif optimization_mode == "Grille de Paramètres":
        # ===== MODE GRILLE =====

        with st.spinner("📊 Génération de la grille..."):
            try:
                # Générer la grille à partir des param_ranges définis par l'utilisateur
                import numpy as np
                from itertools import product
                
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
            st.dataframe(valid_results.head(10), width="stretch")

            # Relancer avec le meilleur
            best = valid_results.iloc[0]
            st.info(f"🥇 Meilleure: {best['params']}")

            # Récupérer le dict original depuis le mapping
            best_params = param_combos_map.get(best['params'], {})
            result, _ = safe_run_backtest(
                engine, df, strategy_key, best_params, symbol, timeframe
            )
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
        
        # Créer l'optimiseur
        with st.spinner("🔌 Connexion au LLM..."):
            try:
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
                show_status("error", f"Échec connexion LLM: {e}")
                st.code(traceback.format_exc())
                st.session_state.is_running = False
                st.stop()
        
        # Exécuter l'optimisation
        st.markdown("---")
        st.markdown("### 📊 Progression de l'optimisation")
        
        # Afficher les logs d'orchestration en temps réel
        st.markdown("#### 📋 Logs d'orchestration")
        orchestration_placeholder = st.empty()

        # Créer le moniteur de progression (LLM: on suit les itérations)
        max_iterations = min(llm_max_iterations, max_combos)
        llm_monitor_placeholder = st.empty()

        # Créer deux colonnes: log itérations + stream pensées
        col_logs, col_thinking = st.columns([1, 1])

        with col_logs:
            iteration_log = st.expander("📝 Log des itérations", expanded=True)

        with col_thinking:
            # Créer viewer pour pensées LLM
            thinking_viewer = ThinkingStreamViewer(container_key="llm_thinking")
            thinking_placeholder = st.empty()

        try:
            with st.spinner("🧠 Optimisation en cours..."):
                # Exemple de pensées pour démonstration
                # Ces pensées seront remplacées par les vraies pensées des agents
                # une fois que le callback sera connecté
                thinking_viewer.add_thought(
                    agent_name="System",
                    model="optimisation",
                    thought="Initialisation de l'optimisation autonome...",
                    category="thinking"
                )

                # Informer l'utilisateur de la configuration
                st.caption(f"🔧 Limite: {max_combos:,} backtests max, {n_workers} workers, {max_iterations} itérations max")

                # Lancer l'optimisation autonome avec limite
                session = strategist.optimize(
                    executor=executor,
                    initial_params=params,
                    param_bounds=param_bounds,
                    max_iterations=max_iterations,  # Limiter par max_combos
                    min_sharpe=-5.0,  # Assouplir contraintes pour permettre exploration même avec baseline négatif
                    max_drawdown=0.50,  # Autoriser plus de drawdown en exploration
                )

                thinking_viewer.add_thought(
                    agent_name="System",
                    model="optimisation",
                    thought=f"Optimisation terminée en {session.current_iteration} itérations",
                    category="conclusion"
                )

                # Afficher le résultat final dans le moniteur
                llm_monitor = ProgressMonitor(total_runs=max_iterations)
                llm_monitor.runs_completed = session.current_iteration
                render_progress_monitor(llm_monitor, llm_monitor_placeholder)

                st.success(f"✅ Optimisation terminée en {session.current_iteration} itérations")

                # Afficher l'historique des itérations
                with iteration_log:
                    for i, exp in enumerate(session.all_results):
                        icon = "🟢" if exp.sharpe_ratio > 0 else "🔴"
                        st.markdown(f"""
                        **Itération {i+1}** {icon}
                        - Params: `{exp.request.parameters}`
                        - Sharpe: `{exp.sharpe_ratio:.3f}`
                        - Return: `{exp.total_return:.2%}`
                        """)

                        # Ajouter pensée pour chaque itération
                        thinking_viewer.add_thought(
                            agent_name="Optimizer",
                            model=llm_model,
                            thought=f"Itération {i+1}: Sharpe={exp.sharpe_ratio:.3f}, Params={exp.request.parameters}",
                            category="thinking"
                        )

                # Afficher le stream de pensées
                with thinking_placeholder:
                    thinking_viewer.render(max_entries=15, show_header=True)
                
                # Sauvegarder et afficher les logs d'orchestration
                orchestration_logger.save_to_file()
                
                # Afficher le viewer complet des logs d'orchestration
                with orchestration_placeholder:
                    st.markdown("---")
                    render_full_orchestration_viewer(
                        orchestration_logger=orchestration_logger,
                        max_entries=50
                    )
                
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
                
        except Exception as e:
            show_status("error", f"Erreur optimisation LLM: {e}")
            st.code(traceback.format_exc())
            st.session_state.is_running = False
            st.stop()

    else:
        # Mode non reconnu
        show_status("error", f"Mode non reconnu: {optimization_mode}")
        st.session_state.is_running = False
        st.stop()

    # ============================================================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================================================

    st.header("📊 Résultats du Backtest")

    # --- Métriques principales ---
    col1, col2, col3, col4 = st.columns(4)

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

    # --- Graphique OHLCV avec Trades ---
    st.subheader("📈 Prix et Trades")

    if result is not None and not result.trades.empty:
        # Afficher le graphique de bougies avec les points d'entrée/sortie
        render_ohlcv_with_trades(
            df=df,
            trades_df=result.trades,
            title="📊 Graphique OHLCV avec Points d'Entrée/Sortie",
            key="ohlcv_trades_main",
            height=600,
        )
    elif result is not None:
        st.info("ℹ️ Aucun trade exécuté, affichage du graphique de prix uniquement")
        render_ohlcv_with_trades(
            df=df,
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

    # Réinitialiser le flag d'exécution à la fin
    st.session_state.is_running = False

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
