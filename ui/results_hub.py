"""
Module-ID: ui.results_hub

Purpose: Vue centralisee des resultats (backtests, sweeps, grids, runs LLM).

Role in pipeline: reporting / catalog

Key components: render_results_hub

Inputs: catalogues CSV, session_state last run

Outputs: Page Streamlit avec dernier run + catalogue filtrable

Dependencies: pandas, streamlit, backtest.storage, utils.run_tracker

Conventions: Non-destructif, lecture des catalogues CSV

Read-if: Ajout d'une page de synthese des resultats.

Skip-if: Vous utilisez seulement ui.results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from backtest.storage import ResultStorage
from ui.helpers import compute_period_days, format_pnl_with_daily
from utils.run_tracker import RunTracker

RESULTS_DIR = Path("backtest_results")
RUNS_DIR = Path("runs")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_catalogs(refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if refresh:
        if RESULTS_DIR.exists():
            storage = ResultStorage(storage_dir=RESULTS_DIR, auto_save=False)
            storage.build_catalogs()
        if RUNS_DIR.exists():
            tracker = RunTracker(cache_file=RUNS_DIR / ".run_cache.json")
            tracker.build_catalogs()

    backtest_overview = _safe_read_csv(RESULTS_DIR / "_catalog" / "overview.csv")
    runs_overview = _safe_read_csv(RUNS_DIR / "_catalog" / "overview.csv")

    backtest_overview = _coerce_numeric(
        backtest_overview,
        [
            "total_pnl",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "total_trades",
            "n_bars",
            "n_trades",
            "n_completed",
            "n_failed",
            "n_trials",
            "n_pruned",
            "best_value",
            "total_time_sec",
            "total_combinations",
            "max_combos",
            "n_workers",
        ],
    )
    runs_overview = _coerce_numeric(
        runs_overview,
        [
            "total_iterations",
            "total_llm_tokens",
            "total_llm_calls",
            "iteration_history_count",
        ],
    )
    return backtest_overview, runs_overview


def _path_to_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except Exception:
        return str(path)


def _add_open_links_backtest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "path" not in df.columns:
        return df
    df = df.copy()
    df["open_folder"] = df["path"].apply(
        lambda value: ""
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == ""
        else _path_to_uri(RESULTS_DIR / str(value))
    )
    return df


def _add_open_links_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    def _row_to_uri(row: pd.Series) -> str:
        trace_path = row.get("trace_path", "")
        if isinstance(trace_path, str) and trace_path:
            return _path_to_uri(Path(trace_path).parent)
        session_id = row.get("session_id", "")
        if session_id:
            return _path_to_uri(RUNS_DIR / str(session_id))
        return ""

    df["open_folder"] = df.apply(_row_to_uri, axis=1)
    return df


def _add_pnl_per_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "total_pnl" not in df.columns:
        return df
    if "period_start" not in df.columns or "period_end" not in df.columns:
        return df
    df = df.copy()
    start_dt = pd.to_datetime(df["period_start"], errors="coerce")
    end_dt = pd.to_datetime(df["period_end"], errors="coerce")
    period_days = (end_dt.dt.date - start_dt.dt.date).dt.days
    period_days = period_days.where(period_days > 0)
    df["period_days"] = period_days
    df["pnl_per_day"] = df["total_pnl"] / df["period_days"]
    if "data_coverage_pct" in df.columns:
        coverage = pd.to_numeric(df["data_coverage_pct"], errors="coerce")
        effective_days = period_days * (coverage / 100.0)
        effective_days = effective_days.where(effective_days > 0)
        df["pnl_per_day_covered"] = df["total_pnl"] / effective_days
    return df


def _sort_by_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = []
    if "total_pnl" in df.columns:
        sort_cols.append("total_pnl")
    if "sharpe_ratio" in df.columns:
        sort_cols.append("sharpe_ratio")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return df


def _pick_latest_from_catalogs(
    backtest_overview: pd.DataFrame,
    runs_overview: pd.DataFrame,
) -> Optional[Dict[str, object]]:
    candidates = []

    if not backtest_overview.empty:
        df = backtest_overview.copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_dt"])
        if not df.empty:
            latest = df.sort_values("timestamp_dt", ascending=False).iloc[0]
            candidates.append({
                "source": "backtest_results",
                "kind": latest.get("type", ""),
                "id": latest.get("id", ""),
                "timestamp": latest.get("timestamp", ""),
                "strategy": latest.get("strategy", ""),
                "symbol": latest.get("symbol", ""),
                "timeframe": latest.get("timeframe", ""),
                "period_start": latest.get("period_start", ""),
                "period_end": latest.get("period_end", ""),
                "metrics": {
                    "total_pnl": latest.get("total_pnl", ""),
                    "total_return_pct": latest.get("total_return_pct", ""),
                    "sharpe_ratio": latest.get("sharpe_ratio", ""),
                    "max_drawdown_pct": latest.get("max_drawdown_pct", ""),
                    "win_rate_pct": latest.get("win_rate_pct", ""),
                    "profit_factor": latest.get("profit_factor", ""),
                },
                "path": latest.get("path", ""),
                "timestamp_dt": latest.get("timestamp_dt"),
            })

    if not runs_overview.empty:
        df = runs_overview.copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_dt"])
        if not df.empty:
            latest = df.sort_values("timestamp_dt", ascending=False).iloc[0]
            candidates.append({
                "source": "runs",
                "kind": latest.get("mode", ""),
                "id": latest.get("session_id", ""),
                "timestamp": latest.get("timestamp", ""),
                "strategy": latest.get("strategy_name", ""),
                "symbol": "",
                "timeframe": "",
                "metrics": {
                    "total_iterations": latest.get("total_iterations", ""),
                    "total_llm_tokens": latest.get("total_llm_tokens", ""),
                    "total_llm_calls": latest.get("total_llm_calls", ""),
                    "last_decision": latest.get("last_decision", ""),
                },
                "path": latest.get("trace_path", ""),
                "timestamp_dt": latest.get("timestamp_dt"),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.get("timestamp_dt") or pd.Timestamp.min, reverse=True)
    return candidates[0]


def _render_latest_run(backtest_overview: pd.DataFrame, runs_overview: pd.DataFrame) -> None:
    st.subheader("🕒 Dernier run")

    session_result = st.session_state.get("last_run_result")
    session_meta = st.session_state.get("last_winner_meta")

    if session_result is not None:
        metrics = session_result.metrics
        meta = session_result.meta
        period_days = compute_period_days(
            meta.get("period_start"),
            meta.get("period_end"),
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "PnL",
                format_pnl_with_daily(metrics.get("total_pnl", 0), period_days),
            )
        with col2:
            st.metric("Return", f"{metrics.get('total_return_pct', 0):.1f}%")
        with col3:
            st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.1f}%")

        st.caption(
            f"Run: {meta.get('run_id', 'n/a')} | "
            f"{meta.get('strategy', 'n/a')} | "
            f"{meta.get('symbol', 'n/a')}/{meta.get('timeframe', 'n/a')}"
        )
        if session_meta and isinstance(session_meta, dict):
            st.caption(f"Origine: {session_meta.get('run_id', 'n/a')}")
        return

    latest = _pick_latest_from_catalogs(backtest_overview, runs_overview)
    if latest is None:
        st.info("Aucun run détecté pour le moment.")
        return

    if latest["source"] == "backtest_results":
        col1, col2, col3, col4 = st.columns(4)
        metrics = latest.get("metrics", {})
        period_days = compute_period_days(
            latest.get("period_start"),
            latest.get("period_end"),
        )
        with col1:
            st.metric(
                "PnL",
                format_pnl_with_daily(metrics.get("total_pnl", 0), period_days),
            )
        with col2:
            st.metric("Return", f"{metrics.get('total_return_pct', 0):.1f}%")
        with col3:
            st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        st.caption(
            f"{latest.get('kind', '')} | {latest.get('id', '')} | "
            f"{latest.get('strategy', '')} {latest.get('symbol', '')}/{latest.get('timeframe', '')} | "
            f"{latest.get('timestamp', '')}"
        )
    else:
        st.info("Dernier run LLM (runs/)")
        metrics = latest.get("metrics", {})
        st.caption(
            f"Mode: {latest.get('kind', '')} | Session: {latest.get('id', '')} | "
            f"Stratégie: {latest.get('strategy', '')} | {latest.get('timestamp', '')}"
        )
        if metrics:
            st.caption(
                f"Iter: {metrics.get('total_iterations', 'n/a')} | "
                f"LLM calls: {metrics.get('total_llm_calls', 'n/a')} | "
                f"Tokens: {metrics.get('total_llm_tokens', 'n/a')} | "
                f"Derniere decision: {metrics.get('last_decision', 'n/a')}"
            )


def _render_overview_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    types = sorted([t for t in df.get("type", pd.Series()).dropna().unique().tolist() if t])
    strategies = sorted([s for s in df.get("strategy", pd.Series()).dropna().unique().tolist() if s])
    symbols = sorted([s for s in df.get("symbol", pd.Series()).dropna().unique().tolist() if s])
    timeframes = sorted([t for t in df.get("timeframe", pd.Series()).dropna().unique().tolist() if t])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_types = st.multiselect("Type", options=types, default=types)
    with col2:
        selected_strategies = st.multiselect("Stratégie", options=strategies, default=strategies)
    with col3:
        selected_symbols = st.multiselect("Symbole", options=symbols, default=symbols)
    with col4:
        selected_timeframes = st.multiselect("Timeframe", options=timeframes, default=timeframes)

    if selected_types:
        df = df[df["type"].isin(selected_types)]
    if selected_strategies:
        df = df[df["strategy"].isin(selected_strategies)]
    if selected_symbols:
        df = df[df["symbol"].isin(selected_symbols)]
    if selected_timeframes:
        df = df[df["timeframe"].isin(selected_timeframes)]
    return df


def _render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    numeric_cols = [c for c in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct"] if c in df.columns]
    if not numeric_cols:
        return

    if PLOTLY_AVAILABLE:
        if "total_return_pct" in df.columns:
            fig = px.histogram(df, x="total_return_pct", nbins=30, title="Distribution Return %")
            st.plotly_chart(fig, width="stretch")
        if {"sharpe_ratio", "max_drawdown_pct"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="max_drawdown_pct",
                y="sharpe_ratio",
                color="type" if "type" in df.columns else None,
                title="Sharpe vs Max Drawdown",
                hover_data=["id", "strategy", "symbol", "timeframe"],
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.bar_chart(df["total_return_pct"].dropna(), height=200)


def _get_numeric_column_config() -> Dict[str, Any]:
    """Configuration des colonnes numériques pour tri correct dans st.dataframe."""
    return {
        "open_folder": st.column_config.LinkColumn("Ouvrir dossier", display_text="📂 Ouvrir"),
        "total_pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
        "pnl_per_day": st.column_config.NumberColumn("PnL/jour ($)", format="$%.2f"),
        "pnl_per_day_covered": st.column_config.NumberColumn("PnL/jour (données)", format="$%.2f"),
        "total_return_pct": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
        "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "max_drawdown_pct": st.column_config.NumberColumn("Max DD (%)", format="%.1f%%"),
        "win_rate_pct": st.column_config.NumberColumn("Win Rate (%)", format="%.1f%%"),
        "data_coverage_pct": st.column_config.NumberColumn("Couverture données (%)", format="%.1f%%"),
        "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
        "total_trades": st.column_config.NumberColumn("Trades", format="%d"),
        "n_bars": st.column_config.NumberColumn("Bars", format="%d"),
        "n_trades": st.column_config.NumberColumn("Trades", format="%d"),
        "n_completed": st.column_config.NumberColumn("Complétés", format="%d"),
        "n_failed": st.column_config.NumberColumn("Échecs", format="%d"),
        "n_trials": st.column_config.NumberColumn("Trials", format="%d"),
        "n_pruned": st.column_config.NumberColumn("Prunés", format="%d"),
        "best_value": st.column_config.NumberColumn("Meilleure val.", format="%.4f"),
        "total_time_sec": st.column_config.NumberColumn("Durée (s)", format="%.1f"),
        "total_combinations": st.column_config.NumberColumn("Combinaisons", format="%d"),
        "max_combos": st.column_config.NumberColumn("Max combos", format="%d"),
        "n_workers": st.column_config.NumberColumn("Workers", format="%d"),
        "total_iterations": st.column_config.NumberColumn("Itérations", format="%d"),
        "total_llm_tokens": st.column_config.NumberColumn("Tokens LLM", format="%d"),
        "total_llm_calls": st.column_config.NumberColumn("Appels LLM", format="%d"),
    }


def render_results_hub() -> None:
    st.header("📚 Résultats & Catalogues")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        refresh = st.button("🔄 Rafraîchir catalogues")
    with col_right:
        st.caption("Catalogues CSV non-destructifs basés sur backtest_results/ et runs/.")

    backtest_overview, runs_overview = _load_catalogs(refresh=refresh)
    backtest_overview = _add_open_links_backtest(backtest_overview)
    runs_overview = _add_open_links_runs(runs_overview)
    backtest_overview = _add_pnl_per_day(backtest_overview)

    _render_latest_run(backtest_overview, runs_overview)

    st.markdown("---")
    st.subheader("🗂️ Catalogue global")

    if backtest_overview.empty and runs_overview.empty:
        st.info("Aucun catalogue disponible. Lancez un run puis cliquez sur Rafraîchir catalogues.")
        return

    # Configuration des colonnes numériques pour tri correct
    numeric_col_config = _get_numeric_column_config()

    tabs = st.tabs([
        "Vue d'ensemble",
        "Backtests",
        "Sweeps",
        "Grids",
        "Optuna",
        "Runs LLM",
        "Catégories",
        "Comparaison",
    ])

    with tabs[0]:
        df = backtest_overview.copy()
        if df.empty:
            st.info("Aucun résultat backtest/sweep/grid.")
        else:
            df = _render_overview_filters(df)
            df = _sort_by_metrics(df)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )
            _render_charts(df)

    with tabs[1]:
        df = backtest_overview[backtest_overview["type"] == "run"].copy() if not backtest_overview.empty else pd.DataFrame()
        if df.empty:
            st.info("Aucun backtest enregistré.")
        else:
            df = _sort_by_metrics(df)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )

    with tabs[2]:
        df = backtest_overview[backtest_overview["type"] == "sweep"].copy() if not backtest_overview.empty else pd.DataFrame()
        if df.empty:
            st.info("Aucun sweep enregistré.")
        else:
            df = _sort_by_metrics(df)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )

    with tabs[3]:
        df = backtest_overview[backtest_overview["type"] == "grid"].copy() if not backtest_overview.empty else pd.DataFrame()
        if df.empty:
            st.info("Aucun grid enregistré.")
        else:
            df = _sort_by_metrics(df)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )

    with tabs[4]:
        df = backtest_overview[backtest_overview["type"] == "optuna"].copy() if not backtest_overview.empty else pd.DataFrame()
        if df.empty:
            st.info("Aucun Optuna enregistré.")
        else:
            df = _sort_by_metrics(df)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )

    with tabs[5]:
        if runs_overview.empty:
            st.info("Aucun run LLM enregistré.")
        else:
            df = runs_overview.copy()
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp_dt", ascending=False, na_position="last").drop(columns=["timestamp_dt"])
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config=numeric_col_config,
            )

    with tabs[6]:
        if backtest_overview.empty:
            st.info("Aucune donnée pour les catégories.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Par stratégie**")
                st.dataframe(
                    backtest_overview["strategy"].value_counts().rename_axis("strategy").reset_index(name="count"),
                    width="stretch",
                    hide_index=True,
                )
            with col2:
                st.markdown("**Par symbole**")
                st.dataframe(
                    backtest_overview["symbol"].value_counts().rename_axis("symbol").reset_index(name="count"),
                    width="stretch",
                    hide_index=True,
                )
            st.markdown("**Par timeframe**")
            st.dataframe(
                backtest_overview["timeframe"].value_counts().rename_axis("timeframe").reset_index(name="count"),
                width="stretch",
                hide_index=True,
            )

    with tabs[7]:
        df = backtest_overview[backtest_overview["type"] == "run"].copy() if not backtest_overview.empty else pd.DataFrame()
        if df.empty:
            st.info("Aucun backtest disponible pour comparaison.")
        else:
            df = _sort_by_metrics(df)
            options = df["id"].dropna().unique().tolist()
            selected = st.multiselect("Sélectionner des runs", options=options)
            if selected:
                selected_df = df[df["id"].isin(selected)]
                st.dataframe(
                    selected_df,
                    width="stretch",
                    hide_index=True,
                    column_config=numeric_col_config,
                )
