"""
Module-ID: ui.helpers

Purpose: Utilitaires UI - tables strat├®gies markdown, stat calcs, cache streamlit helpers.

Role in pipeline: user interface utilities

Key components: generate_strategies_table(), format_metric(), st_cache wrappers

Inputs: Strategies registry, metric values

Outputs: Markdown tables, formatted strings, cached dataframes

Dependencies: streamlit, pandas, ui.constants, ui.context

Conventions: Cache streamlit TTL; markdown tables sync auto; metric formatting pr├®cision.

Read-if: Modification format output ou stat calculations.

Skip-if: Vous appelez generate_strategies_table().
"""

from __future__ import annotations

# pylint: disable=too-many-lines
import math
import statistics
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ui.constants import (
    PARAM_CONSTRAINTS,
    get_strategy_description,
    get_strategy_display_name,
    get_strategy_type,
)
from ui.context import (
    BACKEND_AVAILABLE,
    ParameterSpec,
    calculate_indicator,
    get_storage,
    get_strategy,
    list_strategies,
    load_ohlcv,
)
from utils.observability import generate_run_id, get_obs_logger


def compute_period_days(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    """
    Calcule le nombre de jours entre deux timestamps.

    Args:
        start_ts: Timestamp de d├®but
        end_ts: Timestamp de fin

    Returns:
        Nombre de jours (entier)
    """
    if pd.isna(start_ts) or pd.isna(end_ts):
        return 0
    delta = end_ts - start_ts
    return max(1, int(delta.total_seconds() / 86400))


def compute_period_days_from_df(df: pd.DataFrame) -> int:
    """
    Calcule le nombre de jours couverts par un DataFrame OHLCV.

    Args:
        df: DataFrame avec index datetime

    Returns:
        Nombre de jours (entier)
    """
    if df is None or df.empty:
        return 0
    return compute_period_days(df.index[0], df.index[-1])


def format_pnl_with_daily(
    pnl: float,
    period_days: int,
    show_plus: bool = False,
    escape_markdown: bool = False,
) -> str:
    """
    Formate un PnL avec son ├®quivalent journalier.

    Args:
        pnl: PnL total
        period_days: Nombre de jours de la p├®riode
        show_plus: Si True, affiche un + devant les valeurs positives

    Returns:
        Cha├«ne format├®e "PnL (PnL/jour/day)"
    """
    if period_days <= 0:
        prefix = "+" if show_plus and pnl > 0 else ""
        result = f"{prefix}${pnl:,.2f}"
        return result.replace("$", "\\$") if escape_markdown else result

    pnl_per_day = pnl / period_days
    prefix = "+" if show_plus and pnl > 0 else ""
    result = f"{prefix}${pnl:,.2f} ({prefix}${pnl_per_day:,.2f}/jour)"
    return result.replace("$", "\\$") if escape_markdown else result


def generate_strategies_table() -> str:
    """
    G├®n├¿re dynamiquement le tableau markdown des strat├®gies disponibles.

    Synchronise automatiquement avec le registre des strat├®gies pour ├®viter
    toute divergence entre la sidebar et la page principale.
    """
    available = list_strategies()

    table_lines = [
        "### Strat├®gies Disponibles",
        "",
        "| Strat├®gie | Type | Description |",
        "|-----------|------|-------------|",
    ]

    for strat_key in sorted(available):
        name = get_strategy_display_name(strat_key)
        stype = get_strategy_type(strat_key)
        desc = get_strategy_description(strat_key) or "Strat├®gie personnalis├®e"
        table_lines.append(f"| **{name}** | {stype} | {desc} |")

    return "\n".join(table_lines)


class ProgressMonitor:
    """
    Moniteur de progression en temps r├®el pour les backtests.

    Calcule la vitesse d'ex├®cution et estime le temps restant en utilisant
    une moyenne glissante sur les 3 derni├¿res secondes.
    """

    def __init__(self, total_runs: int):
        self.total_runs = total_runs
        self.runs_completed = 0
        self.start_time = time.perf_counter()
        self.history = deque(maxlen=3)
        self.last_update_time = self.start_time

    def update(self, runs_completed: int) -> Dict[str, Any]:
        self.runs_completed = runs_completed
        current_time = time.perf_counter()

        self.history.append((current_time, runs_completed))

        if len(self.history) >= 2:
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

        elapsed_time = current_time - self.start_time

        remaining_runs = self.total_runs - runs_completed
        if iteration_speed_per_sec > 0 and remaining_runs > 0:
            time_remaining_sec = remaining_runs / iteration_speed_per_sec
        else:
            time_remaining_sec = 0

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
    Affiche la progression du backtest avec gestion des d├®connexions WebSocket.

    Si le client se d├®connecte (page ferm├®e/rafra├«chie), les erreurs sont
    ignor├®es silencieusement au lieu de polluer les logs.
    """
    try:
        metrics = monitor.update(monitor.runs_completed)

        with placeholder.container():
            st.progress(metrics["progress"])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Progression",
                    f"{metrics['runs_completed']}/{metrics['total_runs']}",
                    f"{metrics['progress']*100:.1f}%",
                )

            with col2:
                st.metric(
                    "Vitesse",
                    f"{metrics['speed_per_sec']:.2f} runs/s",
                    f"{metrics['speed_per_2sec']:.1f} runs/2s",
                )

            with col3:
                elapsed_str = monitor.format_time(metrics["elapsed_time_sec"])
                st.metric("Temps ├®coul├®", elapsed_str)

            with col4:
                remaining_str = monitor.format_time(metrics["time_remaining_sec"])
                st.metric("Temps restant", remaining_str)
    except Exception:
        # Client d├®connect├® (WebSocket ferm├®) - ignorer silencieusement
        pass


def show_status(status_type: str, message: str, details: Optional[str] = None):
    if status_type == "success":
        st.success(f"Ô£à {message}")
    elif status_type == "error":
        st.error(f"ÔØî {message}")
        if details:
            with st.expander("D├®tails de l'erreur"):
                st.code(details)
    elif status_type == "warning":
        st.warning(f"ÔÜá´©Å {message}")
    elif status_type == "info":
        st.info(f"Ôä╣´©Å {message}")


def validate_param(name: str, value: Any) -> Tuple[bool, str]:
    if name not in PARAM_CONSTRAINTS:
        return True, ""

    constraints = PARAM_CONSTRAINTS[name]

    if value < constraints["min"]:
        return False, f"{name} doit ├¬tre ÔëÑ {constraints['min']}"

    if value > constraints["max"]:
        return False, f"{name} doit ├¬tre Ôëñ {constraints['max']}"

    return True, ""


def validate_all_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    for name, value in params.items():
        is_valid, error = validate_param(name, value)
        if not is_valid:
            errors.append(error)

    if "fast_period" in params and "slow_period" in params:
        if params["fast_period"] >= params["slow_period"]:
            errors.append("fast_period doit ├¬tre < slow_period")

    return len(errors) == 0, errors


def apply_versioned_preset(preset: Any, strategy_key: str) -> None:
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
    mode: str = "single",
    spec: Optional[ParameterSpec] = None,
    label: Optional[str] = None,
) -> Any:
    constraints: Dict[str, Any] = {}
    is_int = False
    display_name = label or name

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
            st.sidebar.warning(f"Param├¿tre {name} sans contraintes d├®finies")
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
        if is_int:
            return st.sidebar.slider(
                display_name,
                min_value=int(constraints["min"]),
                max_value=int(constraints["max"]),
                value=int(constraints["default"]),
                step=int(constraints["step"]),
                help=constraints["description"],
                key=unique_key,
            )
        return st.sidebar.slider(
            display_name,
            min_value=float(constraints["min"]),
            max_value=float(constraints["max"]),
            value=float(constraints["default"]),
            step=float(constraints["step"]),
            help=constraints["description"],
            key=unique_key,
        )

    with st.sidebar.expander(f"­ƒôè {display_name}", expanded=False):
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
                    key=f"{unique_key}_min",
                )
            with col2:
                param_max = st.number_input(
                    "Max",
                    min_value=int(constraints["min"]),
                    max_value=int(constraints["max"]),
                    value=int(constraints["max"]),
                    step=1,
                    key=f"{unique_key}_max",
                )
            param_step = st.number_input(
                "Step",
                min_value=1,
                max_value=max(1, (int(constraints["max"]) - int(constraints["min"])) // 2),
                value=int(constraints["step"]),
                step=1,
                key=f"{unique_key}_step",
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
                    key=f"{unique_key}_min",
                )
            with col2:
                param_max = st.number_input(
                    "Max",
                    min_value=float(constraints["min"]),
                    max_value=float(constraints["max"]),
                    value=float(constraints["max"]),
                    step=0.1,
                    format="%.2f",
                    key=f"{unique_key}_max",
                )
            param_step = st.number_input(
                "Step",
                min_value=0.01,
                max_value=max(0.1, (float(constraints["max"]) - float(constraints["min"])) / 2),
                value=float(constraints["step"]),
                step=0.01,
                format="%.2f",
                key=f"{unique_key}_step",
            )

        if param_max > param_min and param_step > 0:
            nb_values = int((param_max - param_min) / param_step) + 1
            st.caption(f"ÔåÆ {nb_values} valeurs ├á tester")
        else:
            nb_values = 1
            st.warning("ÔÜá´©Å Plage invalide")

        return {
            "min": param_min,
            "max": param_max,
            "step": param_step,
            "count": nb_values,
        }


def create_constrained_slider(name: str, granularity: float, key_prefix: str = "") -> Any:
    _ = granularity
    return create_param_range_selector(name, key_prefix, mode="single")


def safe_load_data(
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        df = load_ohlcv(symbol, timeframe, start=start, end=end)

        if df is None or df.empty:
            return None, "ÔØî Donn├®es vides ou fichier non trouv├®"

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"ÔØî Colonnes manquantes: {missing}"

        if not isinstance(df.index, pd.DatetimeIndex):
            return None, "ÔØî L'index n'est pas un DatetimeIndex"

        # Validation plus d├®taill├®e des donn├®es
        nan_count = df.isna().sum().sum()
        total_values = len(df) * len(df.columns)
        nan_pct = (nan_count / total_values) * 100 if total_values > 0 else 0

        if nan_pct > 10:
            return None, f"ÔØî Trop de valeurs NaN ({nan_pct:.1f}%, {nan_count}/{total_values})"

        # Validation coh├®rence OHLC
        invalid_ohlc = ((df['high'] < df['low']) |
                       (df['open'] < df['low']) | (df['open'] > df['high']) |
                       (df['close'] < df['low']) | (df['close'] > df['high'])).sum()

        if invalid_ohlc > 0:
            return None, f"ÔØî Donn├®es OHLC incoh├®rentes ({invalid_ohlc} barres)"

        start_fmt = df.index[0].strftime("%Y-%m-%d %H:%M")
        end_fmt = df.index[-1].strftime("%Y-%m-%d %H:%M")
        quality_msg = f"NaN: {nan_pct:.1f}%" if nan_pct > 0 else "Ô£ô Propre"
        return df, f"Ô£à {len(df)} barres ({start_fmt} ÔåÆ {end_fmt}) - {quality_msg}"

    except FileNotFoundError as e:
        from data.loader import _get_data_dir
        data_dir = _get_data_dir()
        return None, f"­ƒôü Fichier non trouv├®: {symbol}_{timeframe} dans {data_dir}"
    except ValueError as e:
        return None, f"­ƒôè Erreur de donn├®es: {str(e)}"
    except pd.errors.EmptyDataError:
        return None, f"­ƒôä Fichier vide: {symbol}_{timeframe}"
    except pd.errors.ParserError as e:
        return None, f"­ƒöº Erreur format fichier: {str(e)}"
    except Exception as exc:
        import traceback
        tb_summary = traceback.format_exc().split('\n')[-3] if len(traceback.format_exc().split('\n')) > 2 else str(exc)
        return None, f"ÔÜá´©Å Erreur inattendue: {tb_summary}"


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
    from .cache_manager import get_cached_data, cache_data

    # V├®rifier cache d'abord
    cached_df = get_cached_data(symbol, timeframe, start_date, end_date)
    if cached_df is not None:
        # Mise ├á jour session state avec donn├®es cached
        st.session_state["ohlcv_df"] = cached_df
        st.session_state["ohlcv_cache_key"] = _data_cache_key(
            symbol, timeframe, start_date, end_date
        )
        st.session_state["ohlcv_status_msg"] = "­ƒôï Donn├®es du cache (5min TTL)"
        return cached_df, "­ƒôï Donn├®es du cache (5min TTL)"

    # Charger depuis source si pas en cache
    start_str = str(start_date) if start_date else None
    end_str = str(end_date) if end_date else None
    df, msg = safe_load_data(symbol, timeframe, start_str, end_str)
    if df is not None:
        # Mettre en cache les nouvelles donn├®es
        cache_data(symbol, timeframe, start_date, end_date, df)
        st.session_state["ohlcv_df"] = df
        st.session_state["ohlcv_cache_key"] = _data_cache_key(
            symbol, timeframe, start_date, end_date
        )
        st.session_state["ohlcv_status_msg"] = msg
    return df, msg


def _parse_run_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _format_run_timestamp(value: Optional[str]) -> str:
    ts = _parse_run_timestamp(value)
    if ts is None:
        return value or "n/a"
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def _format_run_period(start: Optional[str], end: Optional[str]) -> str:
    start_fmt = _format_run_timestamp(start)
    end_fmt = _format_run_timestamp(end)
    if start_fmt == "n/a" and end_fmt == "n/a":
        return "n/a"
    return f"{start_fmt} -> {end_fmt}"


def _find_saved_run_meta(storage: Any, run_id: str) -> Optional[Any]:
    for meta in storage.list_results():
        if meta.run_id == run_id:
            return meta
    return None


def _build_saved_run_label(meta: Any) -> str:
    period = _format_run_period(meta.period_start, meta.period_end)
    return (
        f"{meta.strategy} | {meta.symbol}/{meta.timeframe} | {period} | {meta.run_id}"
    )


def _save_result_to_storage(storage: Any, result: Optional[Any]) -> Tuple[bool, str]:
    if result is None:
        return False, "No result to save."
    run_id = result.meta.get("run_id") or generate_run_id()
    existing_ids = {meta.run_id for meta in storage.list_results()}
    if run_id in existing_ids:
        return False, f"Run already saved: {run_id}"
    try:
        saved_id = storage.save_result(result, run_id=run_id)
    except Exception as exc:
        return False, f"Save failed: {exc}"
    return True, f"Saved run: {saved_id}"


def _maybe_auto_save_run(result: Optional[Any]) -> None:
    if result is None:
        return
    if not st.session_state.get("auto_save_final_run", False):
        return
    if result.meta.get("loaded_from_storage"):
        return
    if not BACKEND_AVAILABLE:
        return
    try:
        storage = get_storage()
    except Exception as exc:
        st.session_state["saved_runs_status"] = f"Auto-save failed: {exc}"
        return
    saved, msg = _save_result_to_storage(storage, result)
    if msg:
        st.session_state["saved_runs_status"] = msg


def render_saved_runs_panel(
    result: Optional[Any],
    strategy_key: str,
    symbol: str,
    timeframe: str,
) -> None:
    st.sidebar.subheader("Saved runs")
    if not BACKEND_AVAILABLE:
        st.sidebar.info("Saved runs unavailable (backend not available).")
        return
    try:
        storage = get_storage()
    except Exception as exc:
        st.sidebar.error(f"Storage error: {exc}")
        return

    status_msg = st.session_state.pop("saved_runs_status", None)
    if status_msg:
        st.sidebar.info(status_msg)

    if "auto_save_final_run" not in st.session_state:
        st.session_state["auto_save_final_run"] = True

    st.sidebar.checkbox(
        "Auto-save final run",
        key="auto_save_final_run",
    )

    if result is not None:
        if st.sidebar.button("Save current run", key="save_current_run"):
            saved, msg = _save_result_to_storage(storage, result)
            if saved:
                st.sidebar.success(msg)
            else:
                st.sidebar.warning(msg)

    filter_current = st.sidebar.checkbox(
        "Filter to current selection",
        value=True,
        key="saved_runs_filter_current",
    )
    filter_text = st.sidebar.text_input(
        "Filter text",
        value="",
        key="saved_runs_filter_text",
    )

    runs = storage.list_results()
    if filter_current:
        runs = [
            r
            for r in runs
            if r.strategy == strategy_key
            and r.symbol == symbol
            and r.timeframe == timeframe
        ]
    if filter_text:
        filter_l = filter_text.lower()
        runs = [
            r
            for r in runs
            if filter_l in _build_saved_run_label(r).lower()
            or filter_l in r.run_id.lower()
        ]

    if not runs:
        st.sidebar.info("No saved runs.")
        return

    run_ids = [r.run_id for r in runs]
    label_map = {r.run_id: _build_saved_run_label(r) for r in runs}
    if st.session_state.get("saved_runs_selected") not in run_ids:
        st.session_state["saved_runs_selected"] = run_ids[0]
    selected_run_id = st.sidebar.selectbox(
        "Select run",
        options=run_ids,
        format_func=lambda rid: label_map.get(rid, rid),
        key="saved_runs_selected",
    )
    selected_meta = next((r for r in runs if r.run_id == selected_run_id), None)
    if selected_meta is not None:
        period_label = _format_run_period(
            selected_meta.period_start,
            selected_meta.period_end,
        )
        st.sidebar.caption(f"Period: {period_label}")
        st.sidebar.caption(
            f"Trades: {selected_meta.n_trades} | Bars: {selected_meta.n_bars}"
        )
        sharpe = selected_meta.metrics.get("sharpe_ratio", 0)
        ret_pct = selected_meta.metrics.get("total_return_pct", 0)
        max_dd = selected_meta.metrics.get("max_drawdown", 0)
        st.sidebar.caption(
            f"Sharpe: {sharpe:.2f} | Return: {ret_pct:.1f}% | MaxDD: {max_dd:.1f}%"
        )

    load_data = st.sidebar.checkbox(
        "Load data for charts",
        value=True,
        key="saved_runs_load_data",
    )
    if st.sidebar.button("Load selected run", key="load_selected_run"):
        st.session_state["pending_run_load_id"] = selected_run_id
        st.session_state["pending_run_load_data"] = load_data
        st.rerun()


def safe_run_backtest(
    engine: Any,
    df: pd.DataFrame,
    strategy: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    run_id: Optional[str] = None,
    silent_mode: bool = False,
    fast_metrics: bool = False,
) -> Tuple[Optional[Any], str]:
    run_id = run_id or generate_run_id(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
    )
    logger = get_obs_logger("ui.app", run_id=run_id, strategy=strategy, symbol=symbol)

    if not silent_mode:
        logger.info("ui_backtest_start params=%s", params)

    try:
        engine.run_id = run_id
        engine.logger = get_obs_logger("backtest.engine", run_id=run_id)

        result = engine.run(
            df=df,
            strategy=strategy,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            silent_mode=silent_mode,
            fast_metrics=fast_metrics,
        )

        pnl = result.metrics.get("total_pnl", 0)
        sharpe = result.metrics.get("sharpe_ratio", 0)

        if not silent_mode:
            logger.info("ui_backtest_end pnl=%.2f sharpe=%.2f", pnl, sharpe)
        return result, f"Termin├® | P&L: ${pnl:,.2f} | Sharpe: {sharpe:.2f}"

    except ValueError as exc:
        logger.warning("ui_backtest_validation_error error=%s", str(exc))
        return None, f"Param├¿tres invalides: {str(exc)}"
    except Exception as exc:
        logger.error("ui_backtest_error error=%s", str(exc))
        return None, f"Erreur: {str(exc)}\n{traceback.format_exc()}"


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


def _aggregate_metric(values: List[Any], method: str, higher_is_better: bool) -> float:
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
            atr_values = calculate_indicator(
                "atr",
                df,
                {"period": atr_period},
            )
            atr_series = pd.Series(atr_values, index=df.index)
            overlays["bollinger"] = {
                "upper": pd.Series(bb_upper, index=df.index),
                "lower": pd.Series(bb_lower, index=df.index),
                "mid": pd.Series(bb_mid, index=df.index),
                "entry_z": entry_z,
            }
            overlays["atr"] = {
                "atr": atr_series,
                "atr_percentile": atr_percentile,
            }

        elif strategy_key == "bollinger_best_longe_3i":
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            entry_level = float(params.get("entry_level", 0.0))
            sl_level = float(params.get("sl_level", -0.5))
            tp_level = float(params.get("tp_level", 0.85))
            atr_period = int(params.get("atr_period", 14))
            atr_percentile = float(params.get("atr_percentile", 30))

            bb_upper, bb_mid, bb_lower = calculate_indicator(
                "bollinger",
                df,
                {"period": bb_period, "std_dev": bb_std},
            )
            atr_values = calculate_indicator(
                "atr",
                df,
                {"period": atr_period},
            )
            upper = pd.Series(bb_upper, index=df.index)
            lower = pd.Series(bb_lower, index=df.index)
            mid = pd.Series(bb_mid, index=df.index)
            entry_line = lower + entry_level * (upper - lower)
            atr_series = pd.Series(atr_values, index=df.index)
            overlays["bollinger"] = {
                "upper": upper,
                "lower": lower,
                "mid": mid,
                "entry_lower": entry_line,
                "sl_level": sl_level,
                "tp_level": tp_level,
            }
            overlays["atr"] = {
                "atr": atr_series,
                "atr_percentile": atr_percentile,
            }

        elif strategy_key == "bollinger_best_short_3i":
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            entry_level = float(params.get("entry_level", 1.0))
            sl_level = float(params.get("sl_level", 1.5))
            tp_level = float(params.get("tp_level", 0.15))
            atr_period = int(params.get("atr_period", 14))
            atr_percentile = float(params.get("atr_percentile", 30))

            bb_upper, bb_mid, bb_lower = calculate_indicator(
                "bollinger",
                df,
                {"period": bb_period, "std_dev": bb_std},
            )
            atr_values = calculate_indicator(
                "atr",
                df,
                {"period": atr_period},
            )
            upper = pd.Series(bb_upper, index=df.index)
            lower = pd.Series(bb_lower, index=df.index)
            mid = pd.Series(bb_mid, index=df.index)
            entry_line = lower + entry_level * (upper - lower)
            atr_series = pd.Series(atr_values, index=df.index)
            overlays["bollinger"] = {
                "upper": upper,
                "lower": lower,
                "mid": mid,
                "entry_upper": entry_line,
                "sl_level": sl_level,
                "tp_level": tp_level,
            }
            overlays["atr"] = {
                "atr": atr_series,
                "atr_percentile": atr_percentile,
            }

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
            macd_line, signal_line, hist = calculate_indicator(
                "macd",
                df,
                {
                    "fast": fast_period,
                    "slow": slow_period,
                    "signal": signal_period,
                },
            )
            overlays["macd"] = {
                "macd": pd.Series(macd_line, index=df.index),
                "signal": pd.Series(signal_line, index=df.index),
                "hist": pd.Series(hist, index=df.index),
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

        elif strategy_key == "ma_crossover":
            fast_period = int(params.get("fast_period", 10))
            slow_period = int(params.get("slow_period", 30))
            close = df["close"]
            overlays["ma"] = {
                "fast": close.rolling(window=fast_period).mean(),
                "slow": close.rolling(window=slow_period).mean(),
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


def safe_copy_cleanup(logger=None) -> None:
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


# LEGACY CODE - SUPPRIM├ë: Remplac├® par version SweepEngine moderne (ligne ~1306)
# Cette fonction n'├®tait jamais appel├®e (├®cras├®e par d├®finition suivante)
def _legacy_run_sweep_parallel_with_callback_UNUSED(
    df,
    strategy_name,
    param_combinations,
    param_names,
    base_params,
    symbol,
    timeframe,
    initial_capital,
    n_workers,
    period_days,
    fast_metrics,
    silent_mode=True,
    callback=None
):
    """[LEGACY - NON UTILIS├ë] Version du sweep parall├¿le avec callback pour affichage temps r├®el."""
    from performance.parallel import ParallelRunner
    import os

    # Configuration parall├®lisation
    config_dict = {
        "n_workers": n_workers,
        "max_in_flight": n_workers * 3,
        "timeout_per_task": 300.0,
        "continue_on_timeout": True,
        "shared_kwargs": {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_capital": initial_capital,
            "period_days": period_days,
            "fast_metrics": fast_metrics,
            "silent_mode": silent_mode
        }
    }

    runner = ParallelRunner(config_dict)

    # Pr├®parer les t├óches
    tasks = []
    for i, combo in enumerate(param_combinations):
        params = base_params.copy()
        for j, param_name in enumerate(param_names):
            params[param_name] = combo[j]

        tasks.append({
            "df": df,
            "params": params,
            "combo_id": i
        })

    # Fonction de traitement des r├®sultats avec callback
    results = []
    completed = 0
    total = len(tasks)

    def process_result(result):
        nonlocal completed, results
        completed += 1
        if result and "error" not in result:
            results.append(result)

        # Callback pour affichage temps r├®el
        if callback:
            try:
                callback(completed, total, result)
            except Exception:
                # Client d├®connect├® - ignorer l'erreur et continuer le sweep
                pass

    # Ex├®cuter avec callback
    from backtest.worker import run_backtest_worker
    try:
        for task in tasks:
            result = run_backtest_worker(task)
            process_result(result)
    except Exception as e:
        print(f"Erreur sweep parall├¿le: {e}")

    return results


def run_sweep_sequential_with_callback(
    df,
    strategy_name,
    param_combinations,
    param_names,
    base_params,
    symbol,
    timeframe,
    initial_capital,
    period_days,
    fast_metrics,
    silent_mode=True,
    callback=None
):
    """Version du sweep s├®quentiel avec callback pour affichage temps r├®el."""
    from ui.context import get_strategy
    from backtest.engine import BacktestEngine
    import time

    results = []
    total = len(param_combinations)

    for i, combo in enumerate(param_combinations):
        # Construire param├¿tres
        params = base_params.copy()
        for j, param_name in enumerate(param_names):
            params[param_name] = combo[j]

        try:
            # Ex├®cuter backtest
            engine = BacktestEngine(initial_capital=initial_capital)
            strategy = get_strategy(strategy_name)()
            result = engine.run(
                df=df,
                strategy=strategy,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                silent_mode=silent_mode
            )

            # Formater pour compatibilit├® avec worker
            result_dict = {
                "params": params,
                "metrics": result.metrics,
                "meta": result.meta,
                "period_days": period_days,
                "combo_id": i
            }
            results.append(result_dict)

            # Callback pour affichage temps r├®el
            if callback:
                try:
                    callback(i + 1, total, result_dict)
                except Exception:
                    # Client d├®connect├® - ignorer l'erreur et continuer le sweep
                    pass

        except Exception as e:
            error_result = {
                "params": params,
                "error": str(e),
                "combo_id": i
            }
            # Callback m├¬me en cas d'erreur
            if callback:
                try:
                    callback(i + 1, total, error_result)
                except Exception:
                    # Client d├®connect├® - ignorer l'erreur et continuer le sweep
                    pass

    return results


def run_sweep_parallel_with_callback(
    df, strategy, param_grid, initial_capital, n_workers=None, callback=None,
    silent_mode=True, fast_metrics=False, symbol="unknown", timeframe="unknown"
):
    """
    Ex├®cute un sweep en parall├¿le avec callback de progression temps r├®el.

    Utilise SweepEngine moderne avec joblib/loky (plus stable que ProcessPoolExecutor).
    Support cache RAM 100k entries pour performance optimale sur gros sweeps.

    Note: GPU d├®sactiv├® pour sweeps (CPU + cache RAM plus efficace, ├®conomise 10 Go VRAM).
    """
    from backtest.sweep import SweepEngine
    import os

    # D├®sactiver GPU pour sweeps (inutile en multiprocess, ├®conomise 10 Go VRAM + ├®vite yoyo 2060)
    os.environ["BACKTEST_USE_GPU"] = "0"
    os.environ["BACKTEST_GPU_QUEUE_ENABLED"] = "0"

    # R├®duire verbosit├® logs pour gros sweeps (├®viter saturation terminal avec 2.4M logs)
    import logging
    logging.getLogger("backtest.engine").setLevel(logging.WARNING)
    logging.getLogger("backtest.sweep").setLevel(logging.INFO)

    if n_workers is None:
        n_workers = max(1, os.cpu_count() // 2)

    # Initialiser SweepEngine avec cache RAM optimis├® si disponible
    try:
        from config.gpu_config_30gb_ram import get_indicator_cache_config
        indicator_cache_config = get_indicator_cache_config()
    except Exception:
        # Fallback: config cache par d├®faut
        indicator_cache_config = None

    engine = SweepEngine(
        max_workers=n_workers,
        initial_capital=initial_capital,
        auto_save=True
    )

    # Callback wrapper pour mettre ├á jour la progression
    class ProgressTracker:
        def __init__(self):
            self.completed = 0
            self.best_result = None
            self.results = []

    tracker = ProgressTracker()

    # Lancer le sweep avec SweepEngine (plus stable que ProcessPoolExecutor)
    try:
        sweep_results = engine.run_sweep(
            df=df,
            strategy=strategy,
            param_grid=param_grid,
            optimize_for="sharpe_ratio",
            silent_mode=silent_mode,
            fast_metrics=fast_metrics,
            indicator_cache_config=indicator_cache_config
        )

        # Formater r├®sultats pour compatibilit├® avec l'UI
        results = []
        for result in sweep_results.results:
            if result is not None:
                formatted_result = {
                    "params": result.params,
                    "metrics": {
                        "total_pnl": result.metrics.total_pnl,
                        "sharpe_ratio": result.metrics.sharpe_ratio,
                        "win_rate_pct": result.metrics.win_rate_pct,
                        "max_drawdown_pct": result.metrics.max_drawdown_pct,
                        "total_trades": result.metrics.total_trades,
                        "profit_factor": result.metrics.profit_factor,
                        "total_return_pct": result.metrics.total_return_pct,
                    }
                }
                results.append(formatted_result)

                # Callback final avec tous les r├®sultats
                if callback:
                    try:
                        best_result = {
                            "result": formatted_result,
                            "best_pnl": result.metrics.total_pnl
                        }
                        callback(len(results), len(sweep_results.results), best_result)
                    except Exception:
                        # Client d├®connect├® - ignorer l'erreur et continuer le sweep
                        pass
            else:
                results.append(None)

        return results

    except Exception as e:
        print(f"Erreur sweep SweepEngine: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_sweep_sequential_with_callback(
    df, strategy, param_grid, initial_capital, callback=None,
    silent_mode=True, fast_metrics=False
):
    """Ex├®cute un sweep en s├®quentiel avec callback de progression."""
    from utils.config import Config
    from backtest.engine import BacktestEngine

    config = Config(initial_capital=initial_capital)
    engine = BacktestEngine(config=config)

    results = []
    total_combos = len(param_grid)
    best_result = None

    for i, params in enumerate(param_grid):
        try:
            result, _ = safe_run_backtest(
                engine, df, strategy, params,
                "unknown", "unknown",  # Pas besoin de symbol/timeframe ici
                silent_mode=silent_mode,
                fast_metrics=fast_metrics
            )

            if result:
                results.append({"metrics": result.metrics, "params": params})
                pnl = result.metrics.get("total_pnl", 0.0)
                if best_result is None or pnl > best_result.get("best_pnl", float("-inf")):
                    best_result = {"result": result, "best_pnl": pnl}
            else:
                results.append(None)

        except Exception:
            results.append(None)

        # Callback de progression
        if callback:
            try:
                callback(i + 1, total_combos, best_result)
            except Exception:
                # Client d├®connect├® - ignorer l'erreur et continuer le sweep
                pass

    return results
