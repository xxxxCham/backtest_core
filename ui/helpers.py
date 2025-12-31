"""
Module-ID: ui.helpers

Purpose: Utilitaires UI - tables stratégies markdown, stat calcs, cache streamlit helpers.

Role in pipeline: user interface utilities

Key components: generate_strategies_table(), format_metric(), st_cache wrappers

Inputs: Strategies registry, metric values

Outputs: Markdown tables, formatted strings, cached dataframes

Dependencies: streamlit, pandas, ui.constants, ui.context

Conventions: Cache streamlit TTL; markdown tables sync auto; metric formatting précision.

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


def generate_strategies_table() -> str:
    """
    Génère dynamiquement le tableau markdown des stratégies disponibles.

    Synchronise automatiquement avec le registre des stratégies pour éviter
    toute divergence entre la sidebar et la page principale.
    """
    available = list_strategies()

    table_lines = [
        "### Stratégies Disponibles",
        "",
        "| Stratégie | Type | Description |",
        "|-----------|------|-------------|",
    ]

    for strat_key in sorted(available):
        name = get_strategy_display_name(strat_key)
        stype = get_strategy_type(strat_key)
        desc = get_strategy_description(strat_key) or "Stratégie personnalisée"
        table_lines.append(f"| **{name}** | {stype} | {desc} |")

    return "\n".join(table_lines)


class ProgressMonitor:
    """
    Moniteur de progression en temps réel pour les backtests.

    Calcule la vitesse d'exécution et estime le temps restant en utilisant
    une moyenne glissante sur les 3 dernières secondes.
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
            st.metric("Temps écoulé", elapsed_str)

        with col4:
            remaining_str = monitor.format_time(metrics["time_remaining_sec"])
            st.metric("Temps restant", remaining_str)


def show_status(status_type: str, message: str, details: Optional[str] = None):
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
    if name not in PARAM_CONSTRAINTS:
        return True, ""

    constraints = PARAM_CONSTRAINTS[name]

    if value < constraints["min"]:
        return False, f"{name} doit être ≥ {constraints['min']}"

    if value > constraints["max"]:
        return False, f"{name} doit être ≤ {constraints['max']}"

    return True, ""


def validate_all_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    for name, value in params.items():
        is_valid, error = validate_param(name, value)
        if not is_valid:
            errors.append(error)

    if "fast_period" in params and "slow_period" in params:
        if params["fast_period"] >= params["slow_period"]:
            errors.append("fast_period doit être < slow_period")

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
) -> Any:
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
        if is_int:
            return st.sidebar.slider(
                name,
                min_value=int(constraints["min"]),
                max_value=int(constraints["max"]),
                value=int(constraints["default"]),
                step=int(constraints["step"]),
                help=constraints["description"],
                key=unique_key,
            )
        return st.sidebar.slider(
            name,
            min_value=float(constraints["min"]),
            max_value=float(constraints["max"]),
            value=float(constraints["default"]),
            step=float(constraints["step"]),
            help=constraints["description"],
            key=unique_key,
        )

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
            st.caption(f"→ {nb_values} valeurs à tester")
        else:
            nb_values = 1
            st.warning("⚠️ Plage invalide")

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
            return None, "Données vides ou fichier non trouvé"

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Colonnes manquantes: {missing}"

        if not isinstance(df.index, pd.DatetimeIndex):
            return None, "L'index n'est pas un DatetimeIndex"

        nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if nan_pct > 10:
            return None, f"Trop de valeurs NaN ({nan_pct:.1f}%)"

        start_fmt = df.index[0].strftime("%Y-%m-%d")
        end_fmt = df.index[-1].strftime("%Y-%m-%d")
        return df, f"OK: {len(df)} barres ({start_fmt} → {end_fmt})"

    except FileNotFoundError:
        return None, f"Fichier non trouvé: {symbol}_{timeframe}"
    except Exception as exc:
        return None, f"Erreur: {str(exc)}"


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

    st.sidebar.checkbox(
        "Auto-save final run",
        value=False,
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
) -> Tuple[Optional[Any], str]:
    run_id = run_id or generate_run_id()
    logger = get_obs_logger("ui.app", run_id=run_id, strategy=strategy, symbol=symbol)

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
        )

        pnl = result.metrics.get("total_pnl", 0)
        sharpe = result.metrics.get("sharpe_ratio", 0)

        logger.info("ui_backtest_end pnl=%.2f sharpe=%.2f", pnl, sharpe)
        return result, f"Terminé | P&L: ${pnl:,.2f} | Sharpe: {sharpe:.2f}"

    except ValueError as exc:
        logger.warning("ui_backtest_validation_error error=%s", str(exc))
        return None, f"Paramètres invalides: {str(exc)}"
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

        elif strategy_key == "bollinger_atr_v2":
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            bb_stop_factor = float(params.get("bb_stop_factor", 1.0))
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
                "stop_factor": bb_stop_factor,
            }
            overlays["atr"] = {
                "atr": atr_series,
                "atr_percentile": atr_percentile,
            }

        elif strategy_key == "bollinger_atr_v3":
            bb_period = int(params.get("bb_period", 20))
            bb_std = float(params.get("bb_std", 2.0))
            entry_z = float(params.get("entry_z", bb_std))
            k_sl = float(params.get("k_sl", 1.5))
            k_tp = float(params.get("k_tp", 2.0))
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
                "k_sl": k_sl,
                "k_tp": k_tp,
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
