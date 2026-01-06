from __future__ import annotations

# pylint: disable=import-outside-toplevel,too-many-lines
import logging
import time
import traceback
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from ui.components.charts import (
    render_comparison_chart,
    render_ohlcv_with_trades_and_indicators,
    render_strategy_param_diagram,
)
from ui.components.sweep_monitor import (
    SweepMonitor,
    render_sweep_progress,
    render_sweep_summary,
)
from ui.constants import PARAM_CONSTRAINTS
from ui.context import (
    LLM_AVAILABLE,
    LLM_IMPORT_ERROR,
    BacktestEngine,
    LiveOrchestrationViewer,
    OrchestrationLogger,
    compute_search_space_stats,
    create_llm_client,
    create_optimizer_from_engine,
    create_orchestrator_with_backtest,
    generate_session_id,
    get_strategy_param_bounds,
    get_strategy_param_space,
    render_deep_trace_viewer,
    render_full_orchestration_viewer,
)

# Import Optuna optimizer
try:
    from backtest.optuna_optimizer import OptunaOptimizer, OPTUNA_AVAILABLE
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None

logger = logging.getLogger(__name__)
from ui.emergency_stop import execute_emergency_stop, get_emergency_handler
from ui.helpers import (
    ProgressMonitor,
    _maybe_auto_save_run,
    build_indicator_overlays,
    build_strategy_params_for_comparison,
    load_selected_data,
    render_progress_monitor,
    safe_load_data,
    safe_run_backtest,
    show_status,
    summarize_comparison_results,
    validate_all_params,
)
from ui.state import SidebarState
from utils.run_tracker import RunSignature, get_global_tracker

_MP_SHARED_DF = None
_MP_INITIAL_CAPITAL = None
_MP_STRATEGY_KEY = None
_MP_SYMBOL = None
_MP_TIMEFRAME = None
_MP_DEBUG_ENABLED = False
_MP_ENGINE = None


def _init_backtest_worker(
    df: pd.DataFrame,
    initial_capital: float,
    strategy_key: str,
    symbol: str,
    timeframe: str,
    debug_enabled: bool,
) -> None:
    global _MP_SHARED_DF, _MP_INITIAL_CAPITAL, _MP_STRATEGY_KEY
    global _MP_SYMBOL, _MP_TIMEFRAME, _MP_DEBUG_ENABLED, _MP_ENGINE
    _MP_SHARED_DF = df
    _MP_INITIAL_CAPITAL = initial_capital
    _MP_STRATEGY_KEY = strategy_key
    _MP_SYMBOL = symbol
    _MP_TIMEFRAME = timeframe
    _MP_DEBUG_ENABLED = debug_enabled
    _MP_ENGINE = BacktestEngine(initial_capital=initial_capital)


def _run_backtest_multiprocess(args):
    """
    Wrapper picklable pour ProcessPoolExecutor.

    Args:
        args: param_combo ou tuple legacy (param_combo, initial_capital, df, strategy_key, symbol, timeframe, debug_enabled)

    Returns:
        Dict avec résultats du backtest ou erreur
    """
    global _MP_ENGINE

    if isinstance(args, tuple) and len(args) == 7:
        param_combo, initial_capital, df, strategy_key, symbol, timeframe, debug_enabled = args
        engine = BacktestEngine(initial_capital=initial_capital)
    else:
        param_combo = args
        df = _MP_SHARED_DF
        strategy_key = _MP_STRATEGY_KEY
        symbol = _MP_SYMBOL
        timeframe = _MP_TIMEFRAME
        debug_enabled = _MP_DEBUG_ENABLED
        if df is None or strategy_key is None:
            raise ValueError("Worker multiprocess non initialisé (missing shared context).")
        if _MP_ENGINE is None:
            if _MP_INITIAL_CAPITAL is None:
                raise ValueError("Worker multiprocess non initialisé (missing initial_capital).")
            _MP_ENGINE = BacktestEngine(initial_capital=_MP_INITIAL_CAPITAL)
        engine = _MP_ENGINE

    try:
        # Utiliser l'engine initialisé par worker (ou fallback legacy)
        result_i, msg_i = safe_run_backtest(
            engine,
            df,
            strategy_key,
            param_combo,
            symbol,
            timeframe,
            silent_mode=not debug_enabled,
        )

        params_native = {
            k: float(v) if hasattr(v, "item") else v for k, v in param_combo.items()
        }
        params_str = str(params_native)

        if result_i:
            return {
                "params": params_str,
                "params_dict": param_combo,
                "total_pnl": result_i.metrics["total_pnl"],
                "sharpe": result_i.metrics["sharpe_ratio"],
                "max_dd": result_i.metrics["max_drawdown_pct"],
                "win_rate": result_i.metrics["win_rate_pct"],
                "trades": result_i.metrics["total_trades"],
                "profit_factor": result_i.metrics["profit_factor"],
            }
        return {
            "params": params_str,
            "params_dict": param_combo,
            "error": msg_i,
        }
    except Exception as exc:
        params_str = str(param_combo)
        return {
            "params": params_str,
            "params_dict": param_combo,
            "error": str(exc),
        }


def render_controls() -> tuple[bool, Any]:
    st.title("📈 Backtest Core - Moteur Simplifié")

    status_container = st.container()

    st.markdown(
        """
Interface avec validation des paramètres et feedback utilisateur.
Le système de granularité limite le nombre de valeurs testables.
"""
    )

    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    st.markdown("---")
    col_btn1, col_btn2, col_spacer = st.columns([2, 2, 6])

    with col_btn1:
        run_button = st.button(
            "🚀 Lancer le Backtest",
            type="primary",
            disabled=st.session_state.is_running,
            use_container_width=True,
            key="btn_run_backtest",
        )

    with col_btn2:
        stop_button = st.button(
            "⛔ Arrêt d'urgence",
            type="secondary",
            disabled=not st.session_state.is_running,
            use_container_width=True,
            key="btn_stop_backtest",
        )

    if stop_button:
        # Signaler l'arrêt via le handler d'urgence
        handler = get_emergency_handler()
        handler.request_stop()

        st.session_state.stop_requested = True
        st.session_state.is_running = False

        # Exécuter le nettoyage complet via emergency_stop
        with st.spinner("🛑 Arrêt d'urgence en cours..."):
            stats = execute_emergency_stop(st.session_state)

        # Afficher le résumé du nettoyage
        n_cleaned = len(stats.get("components_cleaned", []))
        n_errors = len(stats.get("errors", []))

        if n_errors == 0:
            st.success(f"✅ Arrêt d'urgence complet : {n_cleaned} composants nettoyés")
        else:
            st.warning(f"⚠️ Arrêt avec {n_errors} erreurs : {n_cleaned} composants nettoyés")

        st.info("💡 Système prêt pour un nouveau test")

        # Réinitialiser le flag d'arrêt
        handler.reset_stop()
        st.session_state.stop_requested = False
        st.rerun()

    st.markdown("---")

    return run_button, status_container


def render_setup_previews(state: SidebarState) -> None:
    st.markdown("---")
    st.subheader("Schema indicateurs & parametres")
    if state.strategy_instance is None:
        st.info("Selectionnez une strategie pour afficher le schema.")
    else:
        diagram_params = {
            **state.strategy_instance.default_params,
            **state.params,
        }
        render_strategy_param_diagram(
            state.strategy_key,
            diagram_params,
            key=f"diagram_{state.strategy_key}",
        )

    st.markdown("---")
    st.subheader("Apercu OHLCV + indicateurs")
    preview_df = st.session_state.get("ohlcv_df")
    if preview_df is None:
        st.info("Chargez les donnees pour afficher l'apercu.")
    else:
        preview_overlays = build_indicator_overlays(
            state.strategy_key,
            preview_df,
            state.params,
        )
        render_ohlcv_with_trades_and_indicators(
            df=preview_df,
            trades_df=pd.DataFrame(),
            overlays=preview_overlays,
            active_indicators=state.active_indicators,
            title="OHLCV + indicateurs (apercu)",
            key="ohlcv_indicator_preview",
            height=650,
        )


def render_main(
    state: SidebarState,
    run_button: bool,
    status_container: Any,
) -> None:
    result = st.session_state.get("last_run_result")
    winner_params = st.session_state.get("last_winner_params")
    winner_metrics = st.session_state.get("last_winner_metrics")
    winner_origin = st.session_state.get("last_winner_origin")
    winner_meta = st.session_state.get("last_winner_meta")

    params = state.params
    param_ranges = state.param_ranges
    strategy_key = state.strategy_key
    symbol = state.symbol
    timeframe = state.timeframe
    optimization_mode = state.optimization_mode
    debug_enabled = state.debug_enabled
    max_combos = state.max_combos
    n_workers = state.n_workers

    # Optuna config
    use_optuna = state.use_optuna
    optuna_n_trials = state.optuna_n_trials
    optuna_sampler = state.optuna_sampler
    optuna_pruning = state.optuna_pruning
    optuna_metric = state.optuna_metric
    optuna_early_stop = state.optuna_early_stop

    llm_config = state.llm_config
    llm_model = state.llm_model
    llm_use_multi_agent = state.llm_use_multi_agent
    llm_max_iterations = state.llm_max_iterations
    llm_use_walk_forward = state.llm_use_walk_forward
    llm_unload_during_backtest = state.llm_unload_during_backtest
    llm_compare_enabled = state.llm_compare_enabled
    llm_compare_auto_run = state.llm_compare_auto_run
    llm_compare_strategies = state.llm_compare_strategies
    llm_compare_tokens = state.llm_compare_tokens
    llm_compare_timeframes = state.llm_compare_timeframes
    llm_compare_metric = state.llm_compare_metric
    llm_compare_aggregate = state.llm_compare_aggregate
    llm_compare_max_runs = state.llm_compare_max_runs
    llm_compare_use_preset = state.llm_compare_use_preset
    llm_compare_generate_report = state.llm_compare_generate_report

    if run_button:
        st.session_state.is_running = True
        st.session_state.stop_requested = False
        winner_params = None
        winner_metrics = None
        winner_origin = None
        winner_meta = None

        is_valid, errors = validate_all_params(params)

        if not is_valid:
            with status_container:
                show_status("error", "Paramètres invalides")
                for err in errors:
                    st.error(f"  • {err}")
            st.session_state.is_running = False
            st.stop()

        with st.spinner("📥 Chargement des données..."):
            df = st.session_state.get("ohlcv_df")
            data_msg = st.session_state.get("ohlcv_status_msg", "")

            if df is None:
                df, data_msg = load_selected_data(
                    symbol,
                    timeframe,
                    state.start_date,
                    state.end_date,
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

        engine = BacktestEngine(initial_capital=state.initial_capital)

        if optimization_mode == "Backtest Simple":
            with st.spinner("⚙️ Exécution du backtest..."):
                result, result_msg = safe_run_backtest(
                    engine,
                    df,
                    strategy_key,
                    params,
                    symbol,
                    timeframe,
                    silent_mode=not debug_enabled,
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
            _maybe_auto_save_run(result)

        elif optimization_mode == "Grille de Paramètres":
            # Branche Optuna (bayésien) ou Grille classique (exhaustif)
            if use_optuna:
                # === MODE OPTUNA (BAYESIEN) ===
                if not OPTUNA_AVAILABLE:
                    show_status("error", "Optuna non installé. pip install optuna")
                    st.session_state.is_running = False
                    st.stop()

                st.markdown("### ⚡ Optimisation Bayésienne (Optuna)")

                # Construire l'espace des paramètres pour Optuna
                param_space = {}
                for pname, r in param_ranges.items():
                    pmin, pmax, step = r["min"], r["max"], r["step"]
                    if isinstance(pmin, int) and isinstance(step, int):
                        param_space[pname] = {
                            "type": "int",
                            "low": int(pmin),
                            "high": int(pmax),
                            "step": int(step),
                        }
                    else:
                        param_space[pname] = {
                            "type": "float",
                            "low": float(pmin),
                            "high": float(pmax),
                            "step": float(step),
                        }

                st.info(f"⚡ {optuna_n_trials} trials avec algorithme {optuna_sampler.upper()}")

                # DEBUG: Afficher le param_space utilisé
                with st.expander("🔍 Espace de paramètres (DEBUG)", expanded=False):
                    st.json(param_space)
                    st.caption(f"Paramètres optimisés: {list(param_space.keys())}")

                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                results_placeholder = st.empty()

                try:
                    # Créer la config avec les frais
                    from utils.config import Config
                    optuna_config = Config(
                        fees_bps=state.fees_bps if hasattr(state, 'fees_bps') else 10.0,
                        slippage_bps=state.slippage_bps if hasattr(state, 'slippage_bps') else 5.0,
                    )

                    optimizer = OptunaOptimizer(
                        strategy_name=strategy_key,
                        data=df,
                        param_space=param_space,
                        initial_capital=state.initial_capital,
                        config=optuna_config,  # Ajout de la config avec frais
                        seed=42,
                        early_stop_patience=optuna_early_stop if optuna_early_stop > 0 else None,
                        symbol=symbol,
                        timeframe=timeframe,
                    )

                    # Callback pour mise à jour UI en temps réel
                    # OPTIMISATION: Ne rafraîchir l'UI que tous les N trials pour éviter le ralentissement
                    _last_ui_update = [0]  # Mutable pour closure
                    _ui_update_interval = max(1, optuna_n_trials // 100)  # ~100 updates max

                    def optuna_callback(study, trial):
                        n_completed = len(study.trials)

                        # OPTIMISATION: Skip les updates UI intermédiaires pour les gros runs
                        if n_completed - _last_ui_update[0] < _ui_update_interval and n_completed < optuna_n_trials:
                            return  # Skip cette update
                        _last_ui_update[0] = n_completed

                        pct = n_completed / optuna_n_trials
                        progress_bar.progress(min(pct, 1.0))

                        # Récupérer le meilleur P&L depuis l'optimizer
                        best_pnl = optimizer.best_pnl
                        best_return = optimizer.best_return_pct

                        # Formatage du P&L avec couleur
                        if best_pnl > 0:
                            pnl_str = f"+${best_pnl:,.2f}"
                            pnl_delta = f"+{best_return:.1f}%"
                        elif best_pnl > float("-inf"):
                            pnl_str = f"${best_pnl:,.2f}"
                            pnl_delta = f"{best_return:.1f}%"
                        else:
                            pnl_str = "—"
                            pnl_delta = None

                        status_text.text(f"Trial {n_completed}/{optuna_n_trials} - Best P&L: {pnl_str}")

                        with metrics_placeholder.container():
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Complétés", f"{n_completed}/{optuna_n_trials}")
                            # Afficher le meilleur P&L au lieu du Sharpe (qui est 0 pour les comptes ruinés)
                            c2.metric("💰 Meilleur P&L", pnl_str, delta=pnl_delta)
                            n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
                            c3.metric("Pruned", f"{n_pruned}")

                    result_optuna = optimizer.optimize(
                        n_trials=optuna_n_trials,
                        metric=optuna_metric,
                        direction="maximize",
                        sampler=optuna_sampler,
                        pruner="median" if optuna_pruning else "none",
                        callbacks=[optuna_callback],
                        show_progress=False,
                        early_stop_patience=optuna_early_stop if optuna_early_stop > 0 else None,
                    )

                    progress_bar.progress(1.0)
                    # Message de fin avec le meilleur P&L
                    final_pnl = optimizer.best_pnl
                    final_return = optimizer.best_return_pct
                    if final_pnl > float("-inf"):
                        status_text.text(f"✅ Terminé: {result_optuna.n_completed}/{optuna_n_trials} trials | Best P&L: ${final_pnl:,.2f} ({final_return:+.1f}%)")
                    else:
                        status_text.text(f"✅ Terminé: {result_optuna.n_completed}/{optuna_n_trials} trials")

                    st.markdown("---")
                    st.markdown("### 🏆 Résultats Optuna")

                    st.success(f"**Meilleur {optuna_metric}:** {result_optuna.best_value:.4f}")
                    st.json(result_optuna.best_params)

                    # Top 10 résultats
                    top_df = result_optuna.get_top_n(10)
                    if not top_df.empty:
                        st.subheader("🏆 Top 10 Trials")
                        st.dataframe(top_df, use_container_width=True)

                    # Rerun le meilleur pour avoir le résultat complet
                    best_params = {**params, **result_optuna.best_params}
                    result, _ = safe_run_backtest(
                        engine,
                        df,
                        strategy_key,
                        best_params,
                        symbol,
                        timeframe,
                        silent_mode=not debug_enabled,
                    )

                    if result is not None:
                        winner_params = best_params
                        winner_metrics = result.metrics
                        winner_origin = "optuna"
                        winner_meta = result.meta
                        st.session_state["last_run_result"] = result
                        st.session_state["last_winner_params"] = winner_params
                        st.session_state["last_winner_metrics"] = winner_metrics
                        st.session_state["last_winner_origin"] = winner_origin
                        st.session_state["last_winner_meta"] = winner_meta
                        _maybe_auto_save_run(result)

                    with status_container:
                        show_status("success", f"Optuna: {result_optuna.n_completed} trials terminés")

                except Exception as exc:
                    show_status("error", f"Erreur Optuna: {exc}")
                    st.code(traceback.format_exc())
                    st.session_state.is_running = False
                    st.stop()

                # Optuna terminé avec succès - ne pas continuer vers le sweep classique
                st.session_state.is_running = False

            else:
                # === MODE GRILLE CLASSIQUE (EXHAUSTIF) ===
                with st.spinner("📊 Génération de la grille..."):
                    try:
                        param_grid = []
                        param_names = list(param_ranges.keys())

                        if param_names:
                            param_values_lists = []
                            for pname in param_names:
                                r = param_ranges[pname]
                                pmin, pmax, step = r["min"], r["max"], r["step"]

                                if isinstance(pmin, int) and isinstance(step, int):
                                    values = list(range(int(pmin), int(pmax) + 1, int(step)))
                                else:
                                    values = list(
                                        np.arange(float(pmin), float(pmax) + float(step) / 2, float(step))
                                    )
                                    values = [round(v, 2) for v in values if v <= pmax]

                                param_values_lists.append(values)

                            for combo in product(*param_values_lists):
                                # Fusionner params fixes (UI) avec params variants (grille)
                                param_dict = {**params, **dict(zip(param_names, combo))}
                                param_grid.append(param_dict)
                        else:
                            param_grid = [params.copy()]

                        if len(param_grid) > max_combos:
                            st.warning(
                                f"⚠️ Grille limitée: {len(param_grid):,} → {max_combos:,}"
                            )
                            param_grid = param_grid[:max_combos]

                        show_status("info", f"Grille: {len(param_grid):,} combinaisons")

                        # Estimation du temps pour grands volumes
                        if len(param_grid) > 10000:
                            estimated_time_sec = len(param_grid) / max(n_workers, 1) * 0.05
                            if estimated_time_sec > 3600:
                                time_str = f"{estimated_time_sec/3600:.1f} heures"
                            elif estimated_time_sec > 60:
                                time_str = f"{estimated_time_sec/60:.0f} minutes"
                            else:
                                time_str = f"{estimated_time_sec:.0f} secondes"
                            st.info(f"⏱️ Temps estimé: ~{time_str} avec {n_workers} workers")

                    except Exception as exc:
                        show_status("error", f"Échec génération grille: {exc}")
                        st.session_state.is_running = False
                        st.stop()

                # Initialiser les variables locales directement (pas de session_state complexe)
                results_list = []
                param_combos_map = {}

                sweep_monitor = SweepMonitor(
                    total_combinations=len(param_grid),
                    objectives=["sharpe_ratio", "total_return_pct", "total_pnl", "max_drawdown"],
                    top_k=15,
                )
                sweep_monitor.start()

                monitor = ProgressMonitor(total_runs=len(param_grid))

                start_time = time.time()
                last_render_time = start_time

                st.markdown("### 📊 Progression en temps réel")

                def run_single_backtest(param_combo: Dict[str, Any]):
                    try:
                        result_i, msg_i = safe_run_backtest(
                            engine,
                            df,
                            strategy_key,
                            param_combo,
                            symbol,
                            timeframe,
                            silent_mode=not debug_enabled,
                        )

                        params_native = {
                            k: float(v) if hasattr(v, "item") else v for k, v in param_combo.items()
                        }
                        params_str = str(params_native)

                        if result_i:
                            return {
                                "params": params_str,
                                "params_dict": param_combo,
                                "total_pnl": result_i.metrics["total_pnl"],
                                "sharpe": result_i.metrics["sharpe_ratio"],
                                "max_dd": result_i.metrics["max_drawdown_pct"],
                                "win_rate": result_i.metrics["win_rate_pct"],
                                "trades": result_i.metrics["total_trades"],
                                "profit_factor": result_i.metrics["profit_factor"],
                            }
                        return {
                            "params": params_str,
                            "params_dict": param_combo,
                            "error": msg_i,
                        }
                    except Exception as exc:
                        params_str = str(param_combo)
                        return {
                            "params": params_str,
                            "params_dict": param_combo,
                            "error": str(exc),
                        }

                if n_workers > 1 and len(param_grid) > 1:
                    from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

                    total = len(param_grid)
                    completed = 0

                    # Taille de batch adaptative selon le volume
                    batch_size = min(n_workers * 10, 1000, total)  # Max 1000 par batch
                    n_batches = (total + batch_size - 1) // batch_size

                    logger.info(f"🚀 Démarrage sweep parallèle: {total:,} combinaisons, {n_workers} workers, {n_batches} batches")
                    st.info(f"🚀 Démarrage: {total:,} combinaisons en {n_batches} batches ({batch_size} par batch)")

                    # Utiliser st.status pour le streaming en temps réel
                    with st.status(f"🔄 Initialisation... (batch 1/{n_batches})", expanded=True) as status:
                        # Créer les placeholders dans le status
                        metrics_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        results_placeholder = st.empty()

                        # Message initial pendant le démarrage
                        with metrics_placeholder.container():
                            st.info("⏳ Démarrage des workers... Les premiers résultats arrivent dans quelques secondes.")

                        with ProcessPoolExecutor(
                            max_workers=n_workers,
                            initializer=_init_backtest_worker,
                            initargs=(df, state.initial_capital, strategy_key, symbol, timeframe, debug_enabled),
                        ) as executor:
                            # Traitement par batches pour un meilleur feedback
                            for batch_idx in range(n_batches):
                                # Vérifier arrêt d'urgence
                                if st.session_state.get("stop_requested", False):
                                    logger.warning("🛑 Arrêt demandé")
                                    status.update(label=f"⚠️ Arrêté: {completed:,}/{total:,}", state="error")
                                    break

                                batch_start = batch_idx * batch_size
                                batch_end = min(batch_start + batch_size, total)
                                batch_combos = param_grid[batch_start:batch_end]

                                status.update(label=f"🔄 Batch {batch_idx+1}/{n_batches} ({completed:,}/{total:,})")

                                # Soumettre le batch
                                futures = {
                                    executor.submit(_run_backtest_multiprocess, combo): combo
                                    for combo in batch_combos
                                }

                                # Traiter les résultats du batch
                                for future in as_completed(futures):
                                    if st.session_state.get("stop_requested", False):
                                        for f in futures:
                                            f.cancel()
                                        break

                                    completed += 1
                                    monitor.runs_completed = completed

                                    try:
                                        result = future.result()
                                    except Exception as exc:
                                        result = {"params": "error", "error": str(exc)}

                                    params_str = result.get("params", "")
                                    param_combo = result.get("params_dict", {})
                                    param_combos_map[params_str] = param_combo

                                    if "error" not in result:
                                        pnl = result.get("total_pnl", 0.0)
                                        metrics = {
                                            "sharpe_ratio": result.get("sharpe", 0.0),
                                            "total_pnl": pnl,
                                            "total_return_pct": (pnl / state.initial_capital * 100) if state.initial_capital else 0.0,
                                            "max_drawdown": abs(result.get("max_dd", 0.0)),
                                            "win_rate": result.get("win_rate", 0.0),
                                            "total_trades": result.get("trades", 0),
                                            "profit_factor": result.get("profit_factor", 0.0),
                                        }
                                        sweep_monitor.update(params=param_combo, metrics=metrics)
                                    else:
                                        sweep_monitor.update(params=param_combo, metrics={}, error=True)

                                    result_clean = {k: v for k, v in result.items() if k != "params_dict"}
                                    results_list.append(result_clean)

                                    # Mise à jour UI adaptative (temps réel mais throttled)
                                    current_time = time.time()
                                    if current_time - last_render_time > 0.5 or completed == total:
                                        elapsed = current_time - start_time
                                        last_render_time = current_time  # Update last render timestamp

                                        rate = completed / max(elapsed, 0.001)
                                        remaining = total - completed
                                        eta_sec = remaining / rate if rate > 0 else 0

                                        if eta_sec > 3600:
                                            eta_str = f"{eta_sec/3600:.1f}h"
                                        elif eta_sec > 60:
                                            eta_str = f"{eta_sec/60:.0f}min"
                                        else:
                                            eta_str = f"{eta_sec:.0f}s"

                                        pct = (completed / total) * 100

                                        status.update(label=f"🔄 {completed:,}/{total:,} ({pct:.1f}%) - ETA: {eta_str}")
                                        progress_bar.progress(min(completed / total, 1.0))

                                        with metrics_placeholder.container():
                                            c1, c2, c3, c4 = st.columns(4)
                                            c1.metric("Complétés", f"{completed:,}")
                                            c2.metric("Vitesse", f"{rate:.1f}/s")
                                            c3.metric("ETA", eta_str)
                                            c4.metric("Erreurs", f"{sweep_monitor.stats.errors}")

                                        with results_placeholder.container():
                                            # Afficher les premières erreurs si présentes
                                            if sweep_monitor.stats.errors > 0 and completed <= 5:
                                                last_error = results_list[-1].get("error", "Unknown") if results_list else "Unknown"
                                                st.error(f"⚠️ Erreur backtest: {last_error[:200]}")

                                            top_results = sweep_monitor.get_top_results("sharpe_ratio")
                                            # Si tous les sharpe sont 0, utiliser total_pnl
                                            if top_results and all(r.metrics.get("sharpe_ratio", 0) == 0 for r in top_results[:5]):
                                                top_results = sweep_monitor.get_top_results("total_pnl")
                                                metric_label = "PnL"
                                            else:
                                                metric_label = "Sharpe"
                                            if top_results:
                                                st.markdown(f"**🏆 Top 5 actuels ({metric_label}):**")
                                                for i, res in enumerate(top_results[:5]):
                                                    sharpe = res.metrics.get("sharpe_ratio", 0)
                                                    pnl = res.metrics.get("total_pnl", 0)
                                                    ret = res.metrics.get("total_return_pct", 0)
                                                    pf = res.metrics.get("profit_factor", 0)
                                                    st.caption(f"{i+1}. PnL=${pnl:,.0f} | Sharpe={sharpe:.3f} | Return={ret:.1f}% | PF={pf:.2f}")

                        # Fin du sweep
                        status.update(label=f"✅ Terminé: {completed:,}/{total:,}", state="complete")
                else:
                    # Mode séquentiel avec rendu temps réel
                    progress_placeholder = st.empty()
                    total = len(param_grid)
                    last_render_time = time.perf_counter()

                    for i, param_combo in enumerate(param_grid):
                        # Vérifier si un arrêt d'urgence a été demandé
                        if st.session_state.get("stop_requested", False):
                            logger.warning("🛑 Arrêt demandé - interruption du sweep")
                            show_status("warning", f"Arrêté après {i}/{total} runs")
                            break

                        monitor.runs_completed = i + 1

                        result = run_single_backtest(param_combo)
                        params_str = result.get("params", "")
                        param_combo_result = result.get("params_dict", {})
                        param_combos_map[params_str] = param_combo_result

                        if "error" not in result:
                            pnl = result.get("total_pnl", 0.0)
                            metrics = {
                                "sharpe_ratio": result.get("sharpe", 0.0),
                                "total_pnl": pnl,
                                "total_return_pct": (pnl / state.initial_capital * 100) if state.initial_capital else 0.0,
                                "max_drawdown": abs(result.get("max_dd", 0.0)),
                                "win_rate": result.get("win_rate", 0.0),
                                "total_trades": result.get("trades", 0),
                                "profit_factor": result.get("profit_factor", 0.0),
                            }
                            sweep_monitor.update(params=param_combo_result, metrics=metrics)
                        else:
                            sweep_monitor.update(params=param_combo_result, metrics={}, error=True)

                        result_clean = {
                            k: v for k, v in result.items() if k != "params_dict"
                        }
                        results_list.append(result_clean)

                        # Rafraîchir l'UI toutes les 5 itérations ou 300ms
                        current_time = time.perf_counter()
                        if (i + 1) % 5 == 0 or current_time - last_render_time >= 0.3:
                            with progress_placeholder.container():
                                col1, col2, col3, col4 = st.columns(4)
                                elapsed = time.time() - st.session_state.get("_sweep_start_time", time.time())
                                rate = (i + 1) / max(elapsed, 0.001)

                                with col1:
                                    st.metric("Progression", f"{i + 1}/{total}")
                                with col2:
                                    st.metric("Vitesse", f"{rate:.1f}/s")
                                with col3:
                                    eta = (total - i - 1) / rate if rate > 0 else 0
                                    st.metric("ETA", f"{eta:.0f}s")
                                with col4:
                                    pct = ((i + 1) / max(total, 1)) * 100
                                    st.metric("Complété", f"{pct:.1f}%")

                                st.progress((i + 1) / max(total, 1))

                                render_sweep_progress(
                                    sweep_monitor,
                                    key=f"sweep_seq_{i}",
                                    show_top_results=True,
                                    show_evolution=True,
                                )
                            last_render_time = current_time

                st.markdown("---")
                st.markdown("### 🎯 Résumé de l'Optimisation")

                # Debug: afficher le nombre de résultats
                logger.info(f"📊 Sweep terminé: {len(results_list)} résultats collectés")
                st.write(f"**Debug:** {len(results_list)} résultats collectés")

                render_sweep_summary(sweep_monitor, key="sweep_summary")

                with status_container:
                    show_status("success", f"Optimisation: {len(results_list)} tests")

                results_df = pd.DataFrame(results_list)

                if "trades" in results_df.columns:
                    logger.info("=" * 80)
                    logger.info("🔍 DEBUG GRID SEARCH - Analyse de la colonne 'trades'")
                    logger.info("   Type: %s", results_df["trades"].dtype)
                    logger.info("   Shape: %s", results_df["trades"].shape)
                    logger.info(
                        "   Premières valeurs: %s",
                        results_df["trades"].head(10).tolist(),
                    )
                    logger.info(
                        "   Stats: min=%s, max=%s, mean=%.2f",
                        results_df["trades"].min(),
                        results_df["trades"].max(),
                        results_df["trades"].mean(),
                    )

                    trades_values = results_df["trades"].values
                    fractional = [
                        x for x in trades_values if isinstance(x, float) and not x.is_integer()
                    ]
                    if fractional:
                        logger.warning(
                            "   ⚠️  %s valeurs fractionnaires détectées: %s",
                            len(fractional),
                            fractional[:5],
                        )
                    else:
                        logger.info("   ✅ Toutes les valeurs sont des entiers")
                    logger.info("=" * 80)

                error_column = results_df.get("error")
                if error_column is not None:
                    valid_results = results_df[error_column.isna()]
                else:
                    valid_results = results_df

                if not valid_results.empty:
                    # Trier par sharpe puis par total_pnl (pour départager si sharpe=0)
                    valid_results = valid_results.sort_values(
                        ["sharpe", "total_pnl"], ascending=[False, False]
                    )

                    st.subheader("🏆 Top 10 Combinaisons")

                    with st.expander("🔍 Debug Info - Types de données"):
                        st.text(f"Nombre de résultats: {len(valid_results)}")
                        st.text("Types des colonnes:")
                        st.text(str(valid_results.dtypes))
                        if "trades" in valid_results.columns:
                            st.text("\nStatistiques 'trades':")
                            st.text(f"  Type: {valid_results['trades'].dtype}")
                            st.text(f"  Min: {valid_results['trades'].min()}")
                            st.text(f"  Max: {valid_results['trades'].max()}")
                            st.text(
                                f"  Mean: {valid_results['trades'].mean():.2f}"
                            )

                    st.dataframe(valid_results.head(10), width="stretch")

                    best = valid_results.iloc[0]
                    st.info(f"🥇 Meilleure: {best['params']}")

                    best_params = param_combos_map.get(best["params"], {})
                    result, _ = safe_run_backtest(
                        engine,
                        df,
                        strategy_key,
                        best_params,
                        symbol,
                        timeframe,
                        silent_mode=not debug_enabled,
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
                        _maybe_auto_save_run(result)
                else:
                    show_status("error", "Aucun résultat valide")
                    st.session_state.is_running = False
                    st.stop()

        elif optimization_mode == "🤖 Optimisation LLM":
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

            session_id = generate_session_id()
            orchestration_logger = OrchestrationLogger(session_id=session_id)

            try:
                param_bounds = get_strategy_param_bounds(strategy_key)
                if not param_bounds:
                    param_bounds = {}
                    for pname in params.keys():
                        if pname in PARAM_CONSTRAINTS:
                            c = PARAM_CONSTRAINTS[pname]
                            param_bounds[pname] = (c["min"], c["max"])
            except Exception as exc:
                show_status("warning", f"Bornes par défaut utilisées: {exc}")
                param_bounds = {}
                for pname in params.keys():
                    if pname in PARAM_CONSTRAINTS:
                        c = PARAM_CONSTRAINTS[pname]
                        param_bounds[pname] = (c["min"], c["max"])

            try:
                full_param_space = get_strategy_param_space(strategy_key, include_step=True)
                llm_space_stats = compute_search_space_stats(full_param_space)
            except Exception:
                llm_space_stats = None

            max_iterations = min(llm_max_iterations, max_combos)

            comparison_summary: List[Dict[str, Any]] = []
            should_run_comparison = llm_compare_enabled and (
                llm_compare_auto_run or st.session_state.get("llm_compare_run_now", False)
            )
            if should_run_comparison:
                st.subheader("Comparaison multi-strategies")
                if not llm_compare_strategies:
                    st.warning("Aucune strategie selectionnee pour la comparaison.")
                elif not llm_compare_tokens or not llm_compare_timeframes:
                    st.warning("Selectionnez au moins un token et un timeframe.")
                else:
                    start_str = str(state.start_date) if state.start_date else None
                    end_str = str(state.end_date) if state.end_date else None
                    progress_bar = st.progress(0)
                    comparison_results: List[Dict[str, Any]] = []
                    comparison_errors: List[str] = []
                    data_cache: Dict[tuple[str, str], pd.DataFrame] = {}

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
                                    silent_mode=not debug_enabled,
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
                        st.dataframe(pd.DataFrame(comparison_summary), width="stretch")

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
                                    st.warning("LLM indisponible pour la justification.")
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
                                st.warning(f"Justification LLM indisponible: {exc}")
                    st.session_state["llm_compare_run_now"] = False

            st.subheader("🤖 Optimisation par Agents LLM")

            col_info, col_timeline = st.columns([1, 2])

            with col_info:
                st.markdown(
                    f"""
            **Stratégie:** `{strategy_key}`
            **Paramètres initiaux:** `{params}`
            **Max itérations:** {llm_max_iterations}
            **Walk-Forward:** {'✅' if llm_use_walk_forward else '❌'}
            """
                )

                st.markdown("**Bornes des paramètres:**")
                for pname, (pmin, pmax) in param_bounds.items():
                    st.caption(f"• {pname}: [{pmin}, {pmax}]")

                if llm_space_stats:
                    st.markdown("---")
                    if llm_space_stats.is_continuous:
                        st.info("ℹ️ **Espace continu** : exploration adaptative par LLM")
                    else:
                        st.caption(
                            "📊 Espace discret estimé: "
                            f"~{llm_space_stats.total_combinations:,} combinaisons"
                        )
                        st.caption("_(Le LLM explore de façon intelligente sans énumérer)_")

            col_timeline.empty()

            strategist = None
            executor = None
            orchestrator = None

            run_tracker = get_global_tracker()
            data_identifier = (
                f"df_{len(df)}rows_{df.index[0]}_{df.index[-1]}"
                if len(df) > 0
                else "empty_df"
            )
            run_signature = RunSignature(
                strategy_name=strategy_key,
                data_path=data_identifier,
                initial_params=params,
                llm_model=llm_model,
                mode="multi_agents" if llm_use_multi_agent else "autonomous",
                session_id=session_id,
            )

            # Enregistrer le run (pour statistiques) sans bloquer l'exécution
            # Note: Le tracking des duplications durant la session est géré par session_param_tracker
            run_tracker.register(run_signature)

            with st.spinner("🔌 Connexion au LLM..."):
                try:
                    if llm_use_multi_agent:
                        live_events_placeholder = st.empty()
                        live_viewer = LiveOrchestrationViewer(
                            container_key="live_orch_viewer_multi"
                        )

                        def on_orchestration_event(entry):
                            live_viewer.add_event(entry)
                            live_viewer.render(live_events_placeholder, show_header=True)

                        orchestration_logger.set_on_event_callback(on_orchestration_event)

                        orchestrator = create_orchestrator_with_backtest(
                            llm_config=llm_config,
                            strategy_name=strategy_key,
                            data=df,
                            initial_params=params,
                            data_symbol=symbol,
                            data_timeframe=timeframe,
                            role_model_config=state.role_model_config,
                            use_walk_forward=llm_use_walk_forward,
                            orchestration_logger=orchestration_logger,
                            session_id=session_id,
                            n_workers=n_workers,
                            max_iterations=max_iterations,
                            initial_capital=state.initial_capital,
                            config=engine.config,
                        )
                        show_status(
                            "success",
                            "Connexion LLM établie (mode multi-agents)",
                        )
                    else:
                        strategist, executor = create_optimizer_from_engine(
                            llm_config=llm_config,
                            strategy_name=strategy_key,
                            data=df,
                            initial_capital=state.initial_capital,
                            use_walk_forward=llm_use_walk_forward,
                            verbose=True,
                            unload_llm_during_backtest=llm_unload_during_backtest,
                            orchestration_logger=orchestration_logger,
                        )
                        show_status("success", "Connexion LLM établie")
                except Exception as exc:
                    show_status("error", f"Echec connexion LLM: {exc}")
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

                    try:
                        orchestration_logger.save_to_jsonl()
                    except Exception:
                        pass

                    if orchestrator_result.errors:
                        st.warning(
                            f"Orchestration errors: {len(orchestrator_result.errors)}"
                        )
                    if orchestrator_result.warnings:
                        st.warning(
                            f"Orchestration warnings: {len(orchestrator_result.warnings)}"
                        )

                    if orchestrator_result.success:
                        st.success("Optimisation multi-agents terminee")
                    else:
                        st.warning(
                            "Optimisation multi-agents terminee "
                            f"(decision: {orchestrator_result.decision})"
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
                            engine,
                            df,
                            strategy_key,
                            best_params,
                            symbol,
                            timeframe,
                            silent_mode=not debug_enabled,
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
                            _maybe_auto_save_run(result)
                except Exception as exc:
                    show_status("error", f"Erreur optimisation multi-agents: {exc}")
                    st.code(traceback.format_exc())
                    st.session_state.is_running = False
                    st.stop()
            else:
                st.markdown("---")
                st.markdown("### 📊 Progression de l'optimisation LLM")

                live_status = st.status(
                    "🚀 Démarrage de l'optimisation...",
                    expanded=True,
                )
                live_events_placeholder = st.empty()
                orchestration_placeholder = st.empty()

                max_iterations = min(llm_max_iterations, max_combos)

                live_viewer = LiveOrchestrationViewer(
                    container_key="live_orch_viewer"
                )

                def on_orchestration_event(entry):
                    live_viewer.add_event(entry)
                    live_viewer.render(live_events_placeholder, show_header=True)

                orchestration_logger.set_on_event_callback(on_orchestration_event)

                st.caption(
                    "🔧 Limite: "
                    f"{max_combos:,} backtests max, {n_workers} workers, "
                    f"{max_iterations} itérations max"
                )

                try:
                    with live_status:
                        st.write("🤖 **Agent LLM actif** - Optimisation autonome")
                        st.write(
                            f"📊 Stratégie: `{strategy_key}` | Modèle: `{llm_model}`"
                        )

                        session = strategist.optimize(
                            executor=executor,
                            initial_params=params,
                            param_bounds=param_bounds,
                            max_iterations=max_iterations,
                            min_sharpe=-5.0,
                            max_drawdown=0.50,
                        )

                        live_status.update(
                            label=(
                                "✅ Optimisation terminée en "
                                f"{session.current_iteration} itérations"
                            ),
                            state="complete",
                            expanded=False,
                        )

                    st.success(
                        f"✅ Optimisation terminée en {session.current_iteration} itérations"
                    )

                    with st.expander("📝 Historique des itérations", expanded=True):
                        for i, exp in enumerate(session.all_results):
                            icon = "🟢" if exp.sharpe_ratio > 0 else "🔴"
                            col_it1, col_it2, col_it3 = st.columns([2, 1, 1])
                            with col_it1:
                                st.markdown(f"**Itération {i+1}** {icon}")
                                st.caption(
                                    f"Params: `{exp.request.parameters}`"
                                )
                            with col_it2:
                                st.metric("Sharpe", f"{exp.sharpe_ratio:.3f}")
                            with col_it3:
                                st.metric("Return", f"{exp.total_return:.2%}")

                    try:
                        orchestration_logger.save_to_jsonl()
                    except Exception:
                        pass

                    with orchestration_placeholder:
                        st.markdown("---")

                        tab_simple, tab_deep = st.tabs(
                            ["📋 Logs d'orchestration", "🔍 Deep Trace (avancé)"]
                        )

                        with tab_simple:
                            render_full_orchestration_viewer(
                                orchestration_logger=orchestration_logger,
                                max_entries=50,
                            )

                        with tab_deep:
                            if LLM_AVAILABLE:
                                render_deep_trace_viewer(
                                    logger=orchestration_logger
                                )
                            else:
                                st.warning(
                                    "Module LLM non disponible pour Deep Trace avancé"
                                )

                    st.markdown("---")
                    st.subheader("🏆 Résultat de l'optimisation LLM")

                    col_best, col_improve = st.columns(2)

                    with col_best:
                        st.markdown("**Meilleurs paramètres trouvés:**")
                        st.json(session.best_result.request.parameters)

                        st.metric(
                            "Meilleur Sharpe",
                            f"{session.best_result.sharpe_ratio:.3f}",
                        )
                        st.metric(
                            "Return",
                            f"{session.best_result.total_return:.2%}",
                        )

                    with col_improve:
                        if session.all_results:
                            initial_sharpe = session.all_results[0].sharpe_ratio
                            best_sharpe = session.best_result.sharpe_ratio
                            improvement = (
                                (best_sharpe - initial_sharpe) / abs(initial_sharpe) * 100
                            ) if initial_sharpe != 0 else 0

                            st.metric(
                                "Amélioration Sharpe",
                                f"{improvement:+.1f}%",
                                delta=f"{best_sharpe - initial_sharpe:+.3f}",
                            )
                            st.metric("Itérations utilisées", session.current_iteration)

                            if session.final_reasoning:
                                st.info(f"🛑 Arrêt: {session.final_reasoning}")

                    best_params = session.best_result.request.parameters
                    result, _ = safe_run_backtest(
                        engine,
                        df,
                        strategy_key,
                        best_params,
                        symbol,
                        timeframe,
                        silent_mode=not debug_enabled,
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
                        _maybe_auto_save_run(result)

                except Exception as exc:
                    live_status.update(label=f"❌ Erreur: {exc}", state="error")
                    show_status("error", f"Erreur optimisation LLM: {exc}")
                    st.code(traceback.format_exc())
                    st.session_state.is_running = False
                    st.stop()

        else:
            show_status("error", f"Mode non reconnu: {optimization_mode}")
            st.session_state.is_running = False
            st.stop()

    st.session_state.is_running = False
