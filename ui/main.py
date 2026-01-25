"""
UI Streamlit principale pour le moteur de backtest.

PROTECTION WINDOWS SPAWN:
Ce module crée des ProcessPoolExecutor pour les sweeps grille.
Sous Windows, multiprocessing utilise 'spawn' qui ré-exécute le module.
Les workers IMPORTENT ce fichier mais NE DOIVENT PAS exécuter Streamlit.
Protection: Tout code Streamlit est dans main() appelé uniquement par __main__.
"""
from __future__ import annotations

# pylint: disable=import-outside-toplevel,too-many-lines

import gc
import logging
import math
import os
import time
import traceback
from itertools import chain, islice, product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# PERFORMANCE & MEMORY OPTIMIZATIONS
# ==============================================================================

# Cache global pour éviter rechargements répétés
_DATA_CACHE = {}
_CACHE_MAX_SIZE = 10  # Nombre max d'entrées en cache
_CACHE_TTL = 300  # TTL en secondes (5 minutes)

def _get_cached_data(symbol: str, timeframe: str, start_date, end_date) -> Optional[pd.DataFrame]:
    """Récupère les données du cache si disponibles et valides."""
    cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"

    if cache_key in _DATA_CACHE:
        cached_entry = _DATA_CACHE[cache_key]
        # Vérifier TTL
        if time.time() - cached_entry["timestamp"] < _CACHE_TTL:
            return cached_entry["data"].copy()  # Copie défensive
        else:
            # Nettoyer entrée expirée
            del _DATA_CACHE[cache_key]

    return None

def _cache_data(symbol: str, timeframe: str, start_date, end_date, df: pd.DataFrame) -> None:
    """Stocke les données en cache avec nettoyage automatique."""
    cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"

    # Nettoyer le cache si trop plein
    if len(_DATA_CACHE) >= _CACHE_MAX_SIZE:
        # Supprimer l'entrée la plus ancienne
        oldest_key = min(_DATA_CACHE.keys(),
                         key=lambda k: _DATA_CACHE[k]["timestamp"])
        del _DATA_CACHE[oldest_key]
        gc.collect()  # Forcer nettoyage mémoire

    _DATA_CACHE[cache_key] = {
        "data": df.copy(),
        "timestamp": time.time()
    }

def _clear_data_cache() -> None:
    """Nettoie complètement le cache de données."""
    global _DATA_CACHE
    _DATA_CACHE.clear()
    gc.collect()

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

from ui.cache_manager import get_cached_data, cache_data, clear_data_cache
from ui.worker_utils import apply_thread_limit
from ui.llm_handlers import handle_llm_optimization
from ui.constants import PARAM_CONSTRAINTS
from ui.context import (
    BacktestEngine,
    compute_search_space_stats,
    get_strategy_param_bounds,
    get_strategy_param_space,
    render_deep_trace_viewer,
    render_full_orchestration_viewer,
    LiveOrchestrationViewer,
)
from agents.integration import create_comparison_context
from ui.helpers import (
    ProgressMonitor,
    compute_period_days_from_df,
    format_pnl_with_daily,
    render_progress_monitor,
    safe_load_data,
    load_selected_data,
    safe_run_backtest,
    safe_copy_cleanup,
    show_status,
    validate_all_params,
    _maybe_auto_save_run,
)
from ui.state import SidebarState
from ui.components.charts import (
    render_comparison_chart,
    render_multi_sweep_heatmap,
    render_multi_sweep_ranking,
    render_strategy_param_diagram,
    render_ohlcv_with_trades_and_indicators,
)
from ui.components.sweep_monitor import (
    SweepMonitor,
    render_sweep_progress,
    render_sweep_summary,
)
from ui.helpers import build_indicator_overlays
from utils.run_tracker import RunSignature, get_global_tracker

# Import du worker isolé pour éviter les problèmes de pickling avec hot-reload Streamlit
from backtest.worker import run_backtest_worker as _isolated_worker


# ==============================================================================
# UI CONTROLS & ENGINE INITIALIZATION
# ==============================================================================

def render_controls() -> tuple[bool, Any]:
    st.title("📈 Backtest Core - Moteur Simplifié")

    status_container = st.container()

    # Nettoyage périodique du cache (tous les 50 runs)
    if not hasattr(st.session_state, 'cache_cleanup_counter'):
        st.session_state.cache_cleanup_counter = 0

    st.session_state.cache_cleanup_counter += 1
    if st.session_state.cache_cleanup_counter % 50 == 0:
        clear_data_cache()
        st.session_state.cache_cleanup_counter = 0
        st.session_state.cache_cleanup_counter = 0

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
            width="stretch",
            key="btn_run_backtest",
        )

    with col_btn2:
        stop_button = st.button(
            "⛔ Arrêt d'urgence",
            type="secondary",
            disabled=not st.session_state.is_running,
            width="stretch",
            key="btn_stop_backtest",
        )

    if stop_button:
        st.session_state.stop_requested = True
        st.session_state.is_running = False

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                st.success("✅ VRAM GPU vidée")
        except ImportError:
            pass

        logger = logging.getLogger(__name__)
        safe_copy_cleanup(logger)

        st.success("✅ RAM système vidée")
        st.info("💡 Système prêt pour un nouveau test")
        st.session_state.stop_requested = False
        st.rerun()

    st.markdown("---")

    return run_button, status_container


def render_setup_previews(state: SidebarState) -> None:
    """
    Affiche les prévisualisations de configuration (schémas + OHLCV).

    En mode multi-stratégie, organise les graphiques en onglets pour éviter
    un affichage trop chargé.
    """
    from ui.context import get_strategy

    # Récupérer les listes multi-sweep avec fallback sur valeurs simples
    strategy_keys = getattr(state, 'strategy_keys', None)
    if strategy_keys is None:
        strategy_keys = [state.strategy_key] if hasattr(state, 'strategy_key') else []

    all_params = getattr(state, 'all_params', None)
    if all_params is None:
        all_params = {state.strategy_key: state.params} if hasattr(state, 'strategy_key') and hasattr(state, 'params') else {}

    is_multi_strategy = len(strategy_keys) > 1

    st.markdown("---")
    st.subheader("📊 Schema indicateurs & parametres")

    if is_multi_strategy:
        # Mode multi-stratégie : utiliser des onglets
        st.caption(f"💡 {len(strategy_keys)} stratégies sélectionnées - utilisez les onglets ci-dessous")

        tabs = st.tabs([f"📈 {sk}" for sk in strategy_keys])
        for tab, sk in zip(tabs, strategy_keys):
            with tab:
                strat_class = get_strategy(sk)
                if strat_class is None:
                    st.warning(f"Stratégie '{sk}' non trouvée")
                    continue

                # Instancier la classe pour accéder aux properties
                try:
                    strat_instance = strat_class()
                    default_params = strat_instance.default_params or {}
                except Exception:
                    default_params = {}

                strat_params = all_params.get(sk, {})
                diagram_params = {
                    **default_params,
                    **strat_params,
                }
                render_strategy_param_diagram(
                    sk,
                    diagram_params,
                    key=f"diagram_{sk}",
                )
    else:
        # Mode simple : une seule stratégie
        if state.strategy_instance is None:
            st.info("Selectionnez une strategie pour afficher le schema.")
        else:
            default_params = getattr(state.strategy_instance, 'default_params', None) or {}
            current_params = state.params if state.params else {}
            diagram_params = {
                **default_params,
                **current_params,
            }
            render_strategy_param_diagram(
                state.strategy_key,
                diagram_params,
                key=f"diagram_{state.strategy_key}",
            )

    st.markdown("---")
    st.subheader("📉 Apercu OHLCV + indicateurs")

    # En multi-stratégie, afficher un message explicatif
    symbols = getattr(state, 'symbols', [state.symbol])
    timeframes = getattr(state, 'timeframes', [state.timeframe])
    is_multi_data = len(symbols) > 1 or len(timeframes) > 1

    if is_multi_data:
        st.info(
            f"📊 **Multi-sélection active** : {len(symbols)} token(s) × {len(timeframes)} timeframe(s)\n\n"
            f"L'aperçu ci-dessous montre le premier token/timeframe (`{symbols[0]}/{timeframes[0]}`). "
            f"Tous les tokens seront traités lors du sweep."
        )

    preview_df = st.session_state.get("ohlcv_df")
    if preview_df is None:
        st.info("Chargez les donnees pour afficher l'apercu.")
    else:
        # Utiliser les overlays de la première stratégie sélectionnée
        first_strategy = strategy_keys[0]
        first_params = all_params.get(first_strategy, state.params)

        preview_overlays = build_indicator_overlays(
            first_strategy,
            preview_df,
            first_params,
        )
        render_ohlcv_with_trades_and_indicators(
            df=preview_df,
            trades_df=pd.DataFrame(),
            overlays=preview_overlays,
            active_indicators=state.active_indicators,
            title=f"OHLCV + indicateurs ({first_strategy})",
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

    # Multi-sweep: récupérer les listes et dicts
    strategy_keys = state.strategy_keys
    symbols = state.symbols
    timeframes = state.timeframes
    all_params = getattr(state, 'all_params', {strategy_key: params})
    all_param_ranges = getattr(state, 'all_param_ranges', {strategy_key: param_ranges})

    # DEBUG: Tracer les sélections multi-sweep
    if debug_enabled:
        st.info(
            f"🔍 **DEBUG Multi-Sweep**\n\n"
            f"- Strategy keys: {strategy_keys} (len={len(strategy_keys)})\n"
            f"- Symbols: {symbols} (len={len(symbols)})\n"
            f"- Timeframes: {timeframes} (len={len(timeframes)})\n"
            f"- All params keys: {list(all_params.keys())}"
        )

    # Déterminer si on est en mode multi-sweep
    is_multi_sweep = (len(strategy_keys) > 1 or len(symbols) > 1 or len(timeframes) > 1)

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
                # Vérifier d'abord le cache
                df = get_cached_data(symbol, timeframe, state.start_date, state.end_date)
                if df is not None:
                    data_msg = f"{symbol}/{timeframe} (cached) | {len(df):,} barres"
                else:
                    # Charger depuis le disque avec gestion d'erreur
                    try:
                        df, data_msg = load_selected_data(
                            symbol,
                            timeframe,
                            state.start_date,
                            state.end_date,
                        )
                        if df is not None:
                            cache_data(symbol, timeframe, state.start_date, state.end_date, df)
                    except Exception as e:
                        df = None
                        data_msg = f"Erreur: {e}"

            if df is None:
                with status_container:
                    show_status("error", f"Échec chargement: {data_msg}")
                    st.info(
                        "💡 Vérifiez les fichiers dans le répertoire de données configuré"
                    )
                st.session_state.is_running = False
                st.stop()

            if df is not None:
                with status_container:
                    show_status("success", f"Données chargées: {data_msg}")

        period_days = compute_period_days_from_df(df)

        engine = BacktestEngine(initial_capital=state.initial_capital)

        if optimization_mode == "Backtest Simple":
            # Détecter si mode comparaison multi-sweep est demandé
            if is_multi_sweep:
                st.info(
                    f"🔄 **Mode Backtest Simple avec Comparaison Multi-Sweep activé**\n\n"
                    f"- {len(strategy_keys)} stratégie(s): {', '.join(strategy_keys)}\n"
                    f"- {len(symbols)} token(s): {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}\n"
                    f"- {len(timeframes)} timeframe(s): {', '.join(timeframes)}\n\n"
                    f"**Total**: {len(strategy_keys) * len(symbols) * len(timeframes)} backtests"
                )

                # Exécuter les comparaisons multi-sweep avec paramètres fixes
                multi_sweep_results = []
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                total_combinations = len(strategy_keys) * len(symbols) * len(timeframes)
                completed = 0

                for i, strategy_name in enumerate(strategy_keys):
                    # Récupérer les paramètres pour cette stratégie
                    strategy_params = all_params.get(strategy_name, {})

                    if debug_enabled:
                        st.info(f"🔍 **DEBUG**: Traitement stratégie {i+1}/{len(strategy_keys)}: {strategy_name}")

                    for j, symbol in enumerate(symbols):
                        if debug_enabled:
                            st.info(f"🔍 **DEBUG**: Traitement symbol {j+1}/{len(symbols)}: {symbol}")

                        for k, timeframe in enumerate(timeframes):
                            if debug_enabled:
                                st.info(f"🔍 **DEBUG**: Traitement timeframe {k+1}/{len(timeframes)}: {timeframe}")
                                st.info(f"🔍 **DEBUG**: Combinaison {completed+1}/{total_combinations}: {strategy_name} × {symbol} × {timeframe}")
                            try:
                                # Charger les données pour cette combinaison
                                combo_df, msg = safe_load_data(symbol, timeframe,
                                                               str(state.start_date) if state.start_date else None,
                                                               str(state.end_date) if state.end_date else None)

                                if combo_df is None:
                                    continue

                                # Exécuter le backtest avec les paramètres fixes
                                result, result_msg = safe_run_backtest(
                                    engine,
                                    combo_df,
                                    strategy_name,
                                    strategy_params,
                                    symbol,
                                    timeframe,
                                    silent_mode=not debug_enabled,
                                )

                                if result is not None:
                                    # Calculer PnL par jour pour comparaisons
                                    period_days_combo = compute_period_days_from_df(combo_df)
                                    pnl_per_day = result.metrics.get("total_pnl", 0) / max(1, period_days_combo) if period_days_combo else 0

                                    multi_sweep_results.append({
                                        "strategy": strategy_name,
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "total_pnl": result.metrics.get("total_pnl", 0),
                                        "pnl_per_day": pnl_per_day,
                                        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                                        "total_return": result.metrics.get("total_return", 0),
                                        "max_drawdown": result.metrics.get("max_drawdown", 0),
                                        "win_rate": result.metrics.get("win_rate", 0),
                                        "total_trades": result.metrics.get("total_trades", 0),
                                        "period_days": period_days_combo,
                                        "params": strategy_params,
                                        "metrics": result.metrics,
                                    })

                                completed += 1
                                progress_bar.progress(completed / total_combinations)
                                status_placeholder.text(f"Backtest {completed}/{total_combinations}: {strategy_name} × {symbol} × {timeframe}")

                            except Exception as e:
                                st.warning(f"Erreur pour {strategy_name} × {symbol} × {timeframe}: {e}")
                                completed += 1
                                continue

                # Afficher les résultats sous forme de comparaison
                if multi_sweep_results:
                    st.markdown("---")
                    st.subheader("📊 Résultats de Comparaison Multi-Sweep")

                    # Créer le DataFrame des résultats
                    comparison_df = pd.DataFrame(multi_sweep_results)

                    # Trier par PnL décroissant
                    comparison_df = comparison_df.sort_values("total_pnl", ascending=False)

                    # Configuration des colonnes pour affichage
                    from ui.results_hub import _get_numeric_column_config
                    column_config = _get_numeric_column_config()

                    # Tabs pour différentes vues
                    tab_table, tab_heatmap, tab_ranking = st.tabs(["📋 Tableau", "🎯 Heatmap", "🏆 Classement"])

                    with tab_table:
                        # Afficher le tableau principal
                        display_columns = ["strategy", "symbol", "timeframe", "total_pnl", "pnl_per_day",
                                           "sharpe_ratio", "total_return", "max_drawdown", "win_rate", "total_trades"]

                        st.dataframe(
                            comparison_df[display_columns],
                            width="stretch",
                            column_config=column_config,
                            hide_index=True,
                        )

                    with tab_heatmap:
                        # Heatmap interactive stratégie × token/timeframe
                        render_multi_sweep_heatmap(
                            multi_sweep_results,
                            metric="total_pnl",
                            title="Performance par Stratégie × Token/Timeframe",
                            key="backtest_simple_comparison_heatmap",
                        )

                    with tab_ranking:
                        # Classement des meilleurs résultats
                        render_multi_sweep_ranking(
                            multi_sweep_results,
                            metric="total_pnl",
                            top_n=10,
                            title="Top 10 Meilleurs Résultats",
                            key="backtest_simple_comparison_ranking",
                        )

                    # Meilleur global
                    best_result = comparison_df.iloc[0]
                    st.markdown("---")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        pnl_value = best_result["total_pnl"]
                        best_pnl_display = format_pnl_with_daily(
                            best_result["total_pnl"],
                            best_result.get("period_days"),
                            show_plus=True,
                            escape_markdown=True,
                        )
                        message = (
                            f"🏆 **Meilleur résultat**: `{best_result['strategy']}` × "
                            f"`{best_result['symbol']}` × `{best_result['timeframe']}`\n\n"
                            f"💰 PnL: **{best_pnl_display}**"
                        )
                        if pnl_value > 0:
                            st.success(message)
                        elif pnl_value < 0:
                            st.error(message)
                        else:
                            st.info(message)

                    with col2:
                        # Afficher les paramètres utilisés
                        with st.expander("🔧 Paramètres utilisés", expanded=True):
                            for k, v in best_result["params"].items():
                                if not k.startswith("_"):  # Ignorer les params internes
                                    st.text(f"{k}: {v}")

                    # Stocker comme winner pour cohérence avec autres modes
                    winner_params = best_result["params"]
                    winner_metrics = best_result["metrics"]
                    winner_origin = "backtest_comparison"
                    st.session_state["last_winner_params"] = winner_params
                    st.session_state["last_winner_metrics"] = winner_metrics
                    st.session_state["last_winner_origin"] = winner_origin

                    # Créer le contexte de comparaison pour les LLM
                    comparison_summary = [
                        {
                            "strategy": r["strategy"],
                            "symbol": r["symbol"],
                            "timeframe": r["timeframe"],
                            "best_pnl": r["total_pnl"],
                            "sharpe": r["sharpe_ratio"],
                            "best_params": r["params"]
                        }
                        for r in multi_sweep_results[:10]  # Top 10 pour les LLM
                    ]

                    # Stocker pour usage LLM ultérieur avec namespace sécurisé
                    st.session_state["llm_comparison_contexts"] = {
                        "backtest_simple": {
                            "mode": "backtest_simple_comparison",
                            "results": comparison_summary,
                            "total_combinations": len(multi_sweep_results),
                            "strategies_tested": strategy_keys,
                            "tokens_tested": symbols,
                            "timeframes_tested": timeframes,
                            "timestamp": time.time(),  # Pour invalidation cache
                        },
                        "multi_sweep": None  # Placeholder for future multi-sweep context
                    }

                else:
                    st.error("Aucun résultat valide obtenu lors des comparaisons")

                st.session_state.is_running = False
                return  # Sortir après comparaison multi-sweep

            else:
                # Mode Backtest Simple standard (une seule combinaison)
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
            # === MODE MULTI-SWEEP ===
            if is_multi_sweep:
                import gc  # Import une seule fois pour nettoyage mémoire

                total_sweeps = len(strategy_keys) * len(symbols) * len(timeframes)
                st.info(
                    f"🔄 **Mode Multi-Sweep activé**\n\n"
                    f"- {len(strategy_keys)} stratégie(s): {', '.join(strategy_keys)}\n"
                    f"- {len(symbols)} token(s): {', '.join(symbols)}\n"
                    f"- {len(timeframes)} timeframe(s): {', '.join(timeframes)}\n\n"
                    f"➡️ **{total_sweeps} sweep(s)** seront exécutés en série"
                )

                # Accumulateur de résultats multi-sweep
                multi_sweep_results = []
                sweep_idx = 0

                # 🚨 PROTECTION MÉMOIRE MULTI-SWEEP
                # ✅ PRINCIPE FONDAMENTAL : Chaque sweep démarre à 0 en mémoire
                # ✅ AUCUNE limitation basée sur le nombre total de sweeps
                # ✅ Chaque grille peut utiliser ses combinaisons complètes
                # ✅ Nettoyage automatique entre sweeps (gc.collect)
                if total_sweeps > 1:
                    st.success(
                        "🚀 **Mode Multi-Sweep Indépendant**\n\n"
                        "• Chaque sweep utilise sa grille complète\n"
                        "• Mémoire réinitialisée entre sweeps\n"
                        "• Aucune limitation basée sur le nombre de sweeps\n"
                        "• Protection uniquement par sweep individuel"
                    )

                def _estimate_parallel_grid_limit(param_count: int) -> int:
                    try:
                        import psutil
                        available_bytes = psutil.virtual_memory().available
                        per_combo_bytes = 1024 + (param_count * 128)
                        budget_bytes = int(available_bytes * 0.15)  # Plus de mémoire pour sweep individuel
                        estimated_limit = budget_bytes // per_combo_bytes
                        # Limite raisonnable mais généreuse pour sweep unique
                        return max(100_000, estimated_limit)  # Min 100k au lieu de 50k
                    except Exception:
                        return 1_000_000  # Fallback généreux
                try:
                    n_workers_effective = max(1, int(n_workers))
                except (TypeError, ValueError):
                    n_workers_effective = 1
                # Lire threads depuis UI ou fallback env
                try:
                    worker_thread_limit = int(st.session_state.get(
                        "grid_worker_threads",
                        int(os.environ.get("BACKTEST_WORKER_THREADS", "1"))
                    ))
                except (TypeError, ValueError):
                    worker_thread_limit = 1
                worker_thread_limit = max(1, worker_thread_limit)
                apply_thread_limit(worker_thread_limit, label="multi_sweep")

                fast_metrics_env = os.getenv("BACKTEST_SWEEP_FAST_METRICS")
                if fast_metrics_env is not None:
                    fast_metrics_env_value = fast_metrics_env.strip().lower() in ("1", "true", "yes", "on")
                else:
                    fast_metrics_env_value = None
                try:
                    fast_metrics_threshold = int(os.getenv("BACKTEST_SWEEP_FAST_METRICS_THRESHOLD", "500"))
                except (TypeError, ValueError):
                    fast_metrics_threshold = 500
                try:
                    min_parallel_runs = int(os.getenv("BACKTEST_SWEEP_MIN_PARALLEL", "200"))
                except (TypeError, ValueError):
                    min_parallel_runs = 200
                min_parallel_runs = max(0, min_parallel_runs)

                # ═══════════════════════════════════════════════════════════════════════
                # OPTIMISATION CRITIQUE: Pré-charger toutes les données UNIQUES une fois
                # ═══════════════════════════════════════════════════════════════════════
                # AVANT: Charger les données dans la boucle → N_strategies × rechargements
                # APRÈS: Pré-charger combinaisons uniques → 1 chargement par (symbol, tf)
                # Gain: 3 stratégies × même symbol/tf → 3x moins de I/O disque!
                # ═══════════════════════════════════════════════════════════════════════
                sweep_start = state.start_date if state.use_date_filter else None
                sweep_end = state.end_date if state.use_date_filter else None

                # Identifier combinaisons uniques (symbol, timeframe)
                unique_data_keys = set((sym, tf) for sym in symbols for tf in timeframes)

                # Pré-charger toutes les données nécessaires
                st.info(f"🔄 Pré-chargement de {len(unique_data_keys)} dataset(s) unique(s)...")
                preloaded_data = {}
                for sym, tf in unique_data_keys:
                    try:
                        df_preload, msg_preload = load_selected_data(sym, tf, sweep_start, sweep_end)
                        if df_preload is not None and not df_preload.empty:
                            preloaded_data[(sym, tf)] = {
                                "df": df_preload,
                                "msg": msg_preload,
                                "period_days": compute_period_days_from_df(df_preload)
                            }
                            st.success(f"✅ {sym}/{tf}: {len(df_preload):,} barres")
                        else:
                            st.warning(f"⚠️ {sym}/{tf}: {msg_preload}")
                            preloaded_data[(sym, tf)] = None
                    except Exception as e:
                        st.error(f"❌ Erreur chargement {sym}/{tf}: {e}")
                        preloaded_data[(sym, tf)] = None

                st.success(f"✅ Données pré-chargées | {len([v for v in preloaded_data.values() if v])} OK, {len([v for v in preloaded_data.values() if not v])} échecs")

                for sk in strategy_keys:
                    for sym in symbols:
                        for tf in timeframes:
                            sweep_idx += 1

                            # Vérifier si arrêt demandé
                            if st.session_state.get("stop_requested", False):
                                st.warning("⏹️ Arrêt demandé par l'utilisateur")
                                break

                            st.markdown(f"---\n### 🔄 Sweep {sweep_idx}/{total_sweeps}: `{sk}` × `{sym}` × `{tf}`")

                            # Monitoring RAM avant chargement (optionnel)
                            try:
                                import psutil
                                ram_used = psutil.virtual_memory().percent
                                ram_avail_gb = psutil.virtual_memory().available / (1024**3)
                                st.caption(f"💾 RAM: {ram_used:.1f}% utilisée • {ram_avail_gb:.1f} GB disponible")
                            except ImportError:
                                pass

                            # Récupérer les données pré-chargées (ZÉRO I/O disque!)
                            data_entry = preloaded_data.get((sym, tf))
                            if data_entry is None:
                                st.error(f"❌ Pas de données pour {sym}/{tf}")
                                multi_sweep_results.append({
                                    "strategy": sk, "symbol": sym, "timeframe": tf,
                                    "status": "no_data", "best_pnl": None
                                })
                                continue

                            df_sweep = data_entry["df"]
                            load_msg = data_entry["msg"]  # Disponible si besoin de logs
                            period_days_sweep = data_entry["period_days"]

                            # Récupérer les param_ranges pour cette stratégie
                            strat_param_ranges = all_param_ranges.get(sk, {})
                            strat_params = all_params.get(sk, {})

                            if not strat_param_ranges:
                                st.warning(f"⚠️ Pas de paramètres optimisables pour {sk}")
                                multi_sweep_results.append({
                                    "strategy": sk, "symbol": sym, "timeframe": tf,
                                    "status": "no_params", "best_pnl": None
                                })
                                continue

                            # Créer un engine frais pour ce sweep
                            # Générer la grille pour cette stratégie - AVEC PROTECTION ANTI-EXPLOSION
                            param_names_sweep = list(strat_param_ranges.keys())
                            param_values_lists_sweep = []
                            estimated_combos = 1

                            for pname in param_names_sweep:
                                r = strat_param_ranges[pname]
                                pmin, pmax, step = r["min"], r["max"], r["step"]
                                if isinstance(pmin, int) and isinstance(step, int):
                                    values = list(range(int(pmin), int(pmax) + 1, int(step)))
                                else:
                                    values = list(np.arange(float(pmin), float(pmax) + float(step) / 2, float(step)))
                                    values = [round(v, 2) for v in values if v <= pmax]
                                if not values:
                                    values = [pmin]
                                estimated_combos *= len(values)
                                param_values_lists_sweep.append(values)

                            # Définir max_safe_combos pour cette stratégie
                            # Pour multi-sweep, on peut être plus généreux car chaque sweep est indépendant
                            max_combos_env = os.getenv("BACKTEST_SWEEP_MAX_COMBOS")
                            if max_combos_env:
                                try:
                                    max_combos = int(max_combos_env)
                                except (TypeError, ValueError):
                                    max_combos = None
                            else:
                                max_combos = None

                            # � FORCE UTILISATION GRILLE COMPLÈTE EN MULTI-SWEEP
                            # Récupérer max_combos de la sidebar (valeur utilisateur)
                            original_max_combos = max_combos  # Valeur définie par l'utilisateur

                            max_combos_env = os.getenv("BACKTEST_SWEEP_MAX_COMBOS")
                            if max_combos_env:
                                try:
                                    env_max_combos = int(max_combos_env)
                                except (TypeError, ValueError):
                                    env_max_combos = original_max_combos
                            else:
                                env_max_combos = original_max_combos

                            # 🚀 SWEEP INDÉPENDANT : Chaque sweep démarre à 0 en mémoire
                            # Utiliser la grille complète ou la limite utilisateur, JAMAIS de division par nombre de sweeps
                            max_safe_combos = min(env_max_combos, estimated_combos) if env_max_combos is not None else estimated_combos

                            # SOLUTION LAZY : Générer les combinaisons à la volée (pas de stockage en RAM)
                            def generate_combinations_lazy():
                                """Générateur lazy qui produit les combinaisons une par une."""
                                if not param_names_sweep:
                                    yield strat_params.copy()
                                    return

                                combo_count = 0
                                for combo in product(*param_values_lists_sweep):
                                    yield {**strat_params, **dict(zip(param_names_sweep, combo))}
                                    combo_count += 1
                                    # Limite de sécurité même en lazy
                                    if max_safe_combos and combo_count >= max_safe_combos:
                                        break

                            # Calculer le total théorique pour affichage
                            total_theoretical = min(estimated_combos, max_safe_combos)

                            st.info(f"""
                            🚀 **Sweep {sk} - {sym} {tf}**
                            **Combinaisons théoriques :** {estimated_combos:,}
                            **Limite appliquée :** {total_theoretical:,}
                            **Mémoire :** Génération lazy (pas de stockage RAM)
                            **Mode :** {'Parallèle' if total_theoretical >= 200 else 'Séquentiel'}
                            """)

                            # EXÉCUTION avec générateur lazy (zéro RAM pour la grille)
                            combo_generator = generate_combinations_lazy()

                            best_pnl_sweep = float("-inf")
                            best_params_sweep = None
                            completed_sweep = 0
                            progress_bar = st.progress(0.0)

                            # Créer un sweep monitor pour ce sweep individuel
                            sweep_monitor_multi = SweepMonitor(
                                total_combinations=total_theoretical,
                                objectives=["total_pnl", "theoretical_pnl", "sharpe_ratio", "total_return_pct", "max_drawdown"],
                                top_k=10,
                            )
                            sweep_monitor_multi.start()
                            sweep_placeholder_multi = st.empty()
                            last_render_time = time.perf_counter()
                            sweep_start_time = time.perf_counter()

                            fast_metrics = (
                                fast_metrics_env_value
                                if fast_metrics_env_value is not None
                                else (total_theoretical >= fast_metrics_threshold and not debug_enabled)
                            )
                            run_parallel_sweep = (
                                n_workers_effective > 1
                                and total_theoretical > 1
                                and (min_parallel_runs == 0 or total_theoretical >= min_parallel_runs)
                            )
                            if run_parallel_sweep:
                                parallel_grid_limit = _estimate_parallel_grid_limit(len(param_names_sweep))
                                if total_theoretical > parallel_grid_limit:
                                    st.info(
                                        f"ℹ️ Grille volumineuse ({total_theoretical:,} > {parallel_grid_limit:,}). "
                                        "Utilisation mode séquentiel optimisé (lazy, 0 RAM)."
                                    )
                                    run_parallel_sweep = False
                                # 🔥 ISOLATION SWEEP : Même en parallèle, chaque sweep est isolé
                                # Le nettoyage mémoire entre sweeps évite l'accumulation
                            st.caption(
                                f"📊 {total_theoretical:,} combinaisons pour ce sweep • "
                                f"mode: {'parallel' if run_parallel_sweep else 'séquentiel'} • "
                                f"workers: {n_workers_effective} • "
                                f"fast_metrics: {'on' if fast_metrics else 'off'}"
                            )

                            if run_parallel_sweep:
                                from ui.helpers import run_sweep_parallel_with_callback

                                def live_callback(current: int, total: int, best_result) -> None:
                                    """Callback pour affichage live pendant sweep parallèle"""
                                    if total > 0:
                                        progress_bar.progress(min(1.0, current / total))

                                    # Afficher monitor en temps réel toutes les 50 runs ou 1 seconde
                                    nonlocal last_render_time
                                    current_time = time.perf_counter()
                                    if current % 50 == 0 or current_time - last_render_time >= 1.0 or current == total:
                                        last_render_time = current_time
                                        elapsed = current_time - sweep_start_time
                                        bt_per_sec = current / elapsed if elapsed > 0 else 0
                                        progress_pct = (current / total * 100) if total > 0 else 0

                                        with sweep_placeholder_multi.container():
                                            # Afficher débit gaming-style
                                            if bt_per_sec > 100:
                                                speed_emoji, speed_color = "🚀", "#00ff80"
                                                speed_label = "TURBO"
                                            elif bt_per_sec > 10:
                                                speed_emoji, speed_color = "⚡", "#ffff00"
                                                speed_label = "FAST"
                                            else:
                                                speed_emoji, speed_color = "🐢", "#ff5555"
                                                speed_label = "SLOW"

                                            st.markdown(
                                                f"""<div style="background: linear-gradient(45deg, #1e1e2e, #313244);
                                                border: 2px solid {speed_color}; border-radius: 10px; padding: 15px;
                                                box-shadow: 0 0 20px {speed_color}40; margin: 10px 0; text-align: center;">
                                                <h3 style="color: {speed_color}; margin: 0; font-family: 'Courier New';">
                                                {speed_emoji} {bt_per_sec:,.1f} bt/s {speed_label}
                                                </h3>
                                                <p style="color: #cdd6f4; margin: 5px 0;">
                                                {current:,}/{total:,} runs ({progress_pct:.1f}%)
                                                </p></div>""",
                                                unsafe_allow_html=True
                                            )

                                            # Afficher top résultats si disponible
                                            render_sweep_progress(
                                                sweep_monitor_multi,
                                                key=f"sweep_progress_multi_{sweep_idx}_{current}",
                                                show_evolution=False,  # Pas de graphiques en multi-sweep
                                                show_top_results=True
                                            )

                            if run_parallel_sweep:
                                from ui.helpers import run_sweep_parallel_with_callback

                                # Passer le générateur directement pour éviter de bloquer l'UI
                                # La nouvelle version utilise une fenêtre glissante qui accepte les générateurs
                                parallel_error = None
                                try:
                                    parallel_result = run_sweep_parallel_with_callback(
                                        df=df_sweep,
                                        strategy=sk,
                                        param_grid=combo_generator,
                                        initial_capital=state.initial_capital,
                                        n_workers=n_workers_effective,
                                        callback=live_callback,
                                        silent_mode=True,
                                        fast_metrics=fast_metrics,
                                        symbol=sym,
                                        timeframe=tf,
                                    )
                                except Exception as exc:
                                    parallel_error = exc
                                    parallel_result = None

                                if parallel_error is not None:
                                    st.warning(
                                        f"⚠️ Parallélisation échouée ({parallel_error}). Repli séquentiel."
                                    )
                                    run_parallel_sweep = False
                                else:
                                    completed_sweep = len(parallel_result)
                                    for result_data in parallel_result:
                                        if not result_data:
                                            continue

                                        metrics = result_data.get("metrics", {})
                                        # PROTECTION CONTRE -inf : Valider PnL
                                        pnl = metrics.get("total_pnl", 0.0)
                                        if not isinstance(pnl, (int, float)) or not np.isfinite(pnl):
                                            pnl = 0.0  # Fallback si -inf/nan

                                        # Mettre à jour le sweep monitor
                                        sweep_monitor_multi.update(
                                            params=result_data.get("params", {}),
                                            metrics={
                                                "total_pnl": pnl,
                                                "theoretical_pnl": metrics.get("theoretical_pnl", 0.0),
                                                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                                                "total_return_pct": metrics.get("total_return_pct", 0.0),
                                                "max_drawdown": abs(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0))),
                                            }
                                        )

                                        if pnl > best_pnl_sweep:
                                            best_pnl_sweep = pnl
                                            best_params_sweep = result_data.get("params", {})

                            if not run_parallel_sweep:
                                from utils.config import Config
                                sweep_config = Config(initial_capital=state.initial_capital)
                                sweep_engine = BacktestEngine(config=sweep_config)

                                # Mode séquentiel : utiliser le générateur lazy directement
                                successful_runs = 0
                                failed_runs = 0

                                for param_combo in combo_generator:
                                    if st.session_state.get("stop_requested", False):
                                        break

                                    try:
                                        result_i, _ = safe_run_backtest(
                                            sweep_engine, df_sweep, sk, param_combo,
                                            sym, tf, silent_mode=True, fast_metrics=fast_metrics
                                        )
                                        if result_i:
                                            successful_runs += 1
                                            m = result_i.metrics
                                            # PROTECTION CONTRE -inf : Valider PnL
                                            pnl = m.get("total_pnl", 0.0)
                                            if not isinstance(pnl, (int, float)) or not np.isfinite(pnl):
                                                pnl = 0.0  # Fallback si -inf/nan
                                                # Log pour debug
                                                if debug_enabled:
                                                    st.caption(f"⚠️ PnL invalide détecté pour {param_combo}: {m.get('total_pnl', 'N/A')} → 0.0")

                                            # Mettre à jour le sweep monitor
                                            sweep_monitor_multi.update(
                                                params=param_combo,
                                                metrics={
                                                    "total_pnl": pnl,
                                                    "theoretical_pnl": m.get("theoretical_pnl", 0.0),
                                                    "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                                                    "total_return_pct": m.get("total_return_pct", 0.0),
                                                    "max_drawdown": abs(m.get("max_drawdown_pct", m.get("max_drawdown", 0.0))),
                                                }
                                            )

                                            if pnl > best_pnl_sweep:
                                                best_pnl_sweep = pnl
                                                best_params_sweep = param_combo.copy()
                                        else:
                                            failed_runs += 1
                                            sweep_monitor_multi.update(params=param_combo, metrics={}, error=True)
                                    except Exception as e:
                                        failed_runs += 1
                                        sweep_monitor_multi.update(params=param_combo, metrics={}, error=True)
                                        if debug_enabled:
                                            st.caption(f"⚠️ Erreur backtest: {str(e)}")

                                    completed_sweep += 1
                                    current_time = time.perf_counter()
                                    # Afficher métriques temps réel tous les 100 runs OU toutes les 2 secondes
                                    if completed_sweep % 100 == 0 or current_time - last_render_time >= 2.0 or completed_sweep >= total_theoretical:
                                        progress_bar.progress(min(1.0, completed_sweep / total_theoretical))
                                        last_render_time = current_time

                                        # Afficher les métriques en temps réel
                                        with sweep_placeholder_multi.container():
                                            elapsed = current_time - sweep_start_time
                                            bt_per_sec = completed_sweep / elapsed if elapsed > 0 else 0
                                            progress_pct = (completed_sweep / total_theoretical * 100) if total_theoretical > 0 else 0
                                            st.info(f"⚡ {completed_sweep}/{total_theoretical} runs ({progress_pct:.1f}%) • 🚀 {bt_per_sec:.1f} bt/s")

                                            render_sweep_progress(
                                                sweep_monitor_multi,
                                                key=f"sweep_progress_multi_{sweep_idx}",
                                                show_evolution=False,  # Pas de graphiques en multi-sweep pour éviter surcharge
                                                show_top_results=True
                                            )

                                # Debug info à la fin du sweep séquentiel
                                if debug_enabled:
                                    st.info(f"📊 Debug sweep: {successful_runs} réussis, {failed_runs} échoués sur {completed_sweep} runs total")

                            progress_bar.empty()
                            sweep_placeholder_multi.empty()

                            # Enregistrer le résultat
                            pnl_display = (
                                format_pnl_with_daily(
                                    best_pnl_sweep,
                                    period_days_sweep,
                                    show_plus=True,
                                    escape_markdown=True,
                                )
                                if best_pnl_sweep > float("-inf")
                                else "N/A"
                            )
                            status = "success" if best_pnl_sweep > float("-inf") else "no_valid_result"
                            best_pnl_per_day = (
                                best_pnl_sweep / period_days_sweep
                                if period_days_sweep and best_pnl_sweep > float("-inf")
                                else None
                            )
                            multi_sweep_results.append({
                                "strategy": sk, "symbol": sym, "timeframe": tf,
                                "status": status, "best_pnl": best_pnl_sweep if best_pnl_sweep > float("-inf") else None,
                                "best_params": best_params_sweep,
                                "best_pnl_per_day": best_pnl_per_day,
                                "period_days": period_days_sweep,
                                "combinations": total_theoretical,
                            })

                            if best_pnl_sweep > float("-inf"):
                                message = f"✅ Meilleur PnL: {pnl_display}"
                                if best_pnl_sweep > 0:
                                    st.success(message)
                                elif best_pnl_sweep < 0:
                                    st.error(message)
                                else:
                                    st.info(message)
                            else:
                                st.warning("⚠️ Aucun résultat valide")

                            # Nettoyage mémoire AGRESSIF après chaque sweep
                            if 'sweep_engine' in locals():
                                del sweep_engine
                            if 'param_grid_sweep' in locals():
                                del param_grid_sweep
                            if 'sweep_monitor_multi' in locals():
                                del sweep_monitor_multi
                            if 'combo_generator' in locals():
                                del combo_generator
                            del df_sweep

                            # Forcer nettoyage immédiat pour libérer RAM entre sweeps
                            gc.collect()

                            # Optionnel : Afficher RAM libre (debug)
                            if debug_enabled:
                                try:
                                    import psutil
                                    ram_avail = psutil.virtual_memory().available / (1024**3)
                                    st.caption(f"🧹 RAM libre après nettoyage: {ram_avail:.1f} GB")
                                except ImportError:
                                    pass

                    if st.session_state.get("stop_requested", False):
                        break

                # === AFFICHAGE DU RÉSUMÉ MULTI-SWEEP ===
                st.markdown("---\n### 📊 Résumé Multi-Sweep")

                if multi_sweep_results:
                    # Onglets pour organiser les différentes vues
                    tab_table, tab_heatmap, tab_ranking = st.tabs([
                        "📋 Tableau", "🗺️ Heatmap", "🏆 Classement"
                    ])

                    with tab_table:
                        # Tableau récapitulatif avec colonnes numériques pour tri correct
                        summary_df = pd.DataFrame(multi_sweep_results)
                        display_df = summary_df.copy()

                        # Configurer les colonnes pour affichage et tri
                        column_config = {
                            "strategy": st.column_config.TextColumn("Stratégie"),
                            "symbol": st.column_config.TextColumn("Token"),
                            "timeframe": st.column_config.TextColumn("TF"),
                            "status": st.column_config.TextColumn("Status"),
                            "best_pnl": st.column_config.NumberColumn(
                                "Meilleur PnL",
                                format="$%.2f",
                                help="PnL du meilleur résultat"
                            ),
                            "best_pnl_per_day": st.column_config.NumberColumn(
                                "PnL/jour",
                                format="$%.2f",
                                help="PnL moyen par jour"
                            ),
                            "combinations": st.column_config.NumberColumn(
                                "Combinaisons",
                                format="%d",
                                help="Nombre de combinaisons testées"
                            ),
                        }

                        st.dataframe(
                            display_df[[
                                "strategy",
                                "symbol",
                                "timeframe",
                                "status",
                                "best_pnl",
                                "best_pnl_per_day",
                                "combinations",
                            ]],
                            width="stretch",
                            column_config=column_config,
                            hide_index=True,
                        )

                    with tab_heatmap:
                        # Heatmap interactive stratégie × token/timeframe
                        render_multi_sweep_heatmap(
                            multi_sweep_results,
                            metric="best_pnl",
                            title="Performance par Stratégie × Token/Timeframe",
                            key="multi_sweep_heatmap_main",
                        )

                    with tab_ranking:
                        # Classement des meilleurs résultats
                        render_multi_sweep_ranking(
                            multi_sweep_results,
                            metric="best_pnl",
                            top_n=10,
                            title="Top 10 Meilleurs Résultats",
                            key="multi_sweep_ranking_main",
                        )

                    # Meilleur global + PnL cumulé (toujours visible)
                    valid_sweeps = [r for r in multi_sweep_results if r.get("best_pnl") is not None]
                    if valid_sweeps:
                        best_overall = max(valid_sweeps, key=lambda r: r["best_pnl"])

                        # Calculer PnL cumulé de tous les sweeps
                        cumulative_pnl = sum(r["best_pnl"] for r in valid_sweeps)
                        avg_period_days = sum(r.get("period_days", 0) for r in valid_sweeps if r.get("period_days")) / max(len([r for r in valid_sweeps if r.get("period_days")]), 1)

                        st.markdown("---")
                        col1, col2, col3 = st.columns([2, 1.5, 1])

                        with col1:
                            best_overall_pnl = format_pnl_with_daily(
                                best_overall["best_pnl"],
                                best_overall.get("period_days"),
                                show_plus=True,
                                escape_markdown=True,
                            )
                            message = (
                                f"🏆 **Meilleur sweep**: `{best_overall['strategy']}` × "
                                f"`{best_overall['symbol']}` × `{best_overall['timeframe']}`\n\n"
                                f"💰 PnL: **{best_overall_pnl}**"
                            )
                            if best_overall["best_pnl"] > 0:
                                st.success(message)
                            elif best_overall["best_pnl"] < 0:
                                st.error(message)
                            else:
                                st.info(message)

                        with col2:
                            # PnL cumulé de tous les sweeps
                            cumul_display = format_pnl_with_daily(
                                cumulative_pnl,
                                avg_period_days,
                                show_plus=True,
                                escape_markdown=True,
                            )
                            message = (
                                f"📊 **PnL Cumulé Total**\n\n"
                                f"({len(valid_sweeps)} sweeps)\n\n"
                                f"💰 **{cumul_display}**"
                            )
                            if cumulative_pnl > 0:
                                st.success(message)
                            elif cumulative_pnl < 0:
                                st.error(message)
                            else:
                                st.info(message)

                        with col3:
                            # Afficher les meilleurs paramètres
                            best_params = best_overall.get("best_params")
                            if best_params:
                                with st.expander("🔧 Paramètres gagnants", expanded=True):
                                    for k, v in best_params.items():
                                        if not k.startswith("_"):  # Ignorer les params internes
                                            st.text(f"{k}: {v}")

                        # Stocker comme winner
                        winner_params = best_overall.get("best_params")
                        winner_origin = "multi_sweep"
                        st.session_state["last_winner_params"] = winner_params
                        st.session_state["last_winner_origin"] = winner_origin

                st.session_state.is_running = False
                return  # Sortir après multi-sweep

            # === MODE SWEEP SIMPLE (code existant) ===
            try:
                n_workers_effective = max(1, int(n_workers))
            except (TypeError, ValueError):
                n_workers_effective = 1
            # Lire threads depuis UI ou fallback env
            try:
                worker_thread_limit = int(st.session_state.get("grid_worker_threads",
                                                                 int(os.environ.get("BACKTEST_WORKER_THREADS", "1"))))
            except (TypeError, ValueError):
                worker_thread_limit = 1
            worker_thread_limit = max(1, worker_thread_limit)
            apply_thread_limit(worker_thread_limit, label="main")

            with st.spinner("📊 Génération de la grille..."):
                try:
                    param_names = list(param_ranges.keys())
                    param_values_lists = []

                    if param_names:
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

                            if not values:
                                values = [pmin]

                            param_values_lists.append(values)

                        total_combinations = max(
                            1, math.prod(len(values) for values in param_values_lists)
                        )
                        combo_iter = (
                            {**params, **dict(zip(param_names, combo))}
                            for combo in product(*param_values_lists)
                        )
                    else:
                        total_combinations = 1
                        combo_iter = iter([params.copy()])

                    # Appliquer limite uniquement si max_combos < 100M (considéré comme illimité au-delà)
                    if max_combos and max_combos < 100_000_000 and total_combinations > max_combos:
                        st.warning(
                            f"⚠️ Grille limitée: {total_combinations:,} → {max_combos:,}"
                        )
                        total_runs = max_combos
                        combo_iter = islice(combo_iter, max_combos)
                    else:
                        total_runs = total_combinations

                    if total_runs < total_combinations:
                        show_status(
                            "info",
                            f"Grille: {total_runs:,} / {total_combinations:,} combinaisons ({n_workers_effective} workers × {worker_thread_limit} threads)",
                        )
                    else:
                        show_status("info", f"Grille: {total_runs:,} combinaisons ({n_workers_effective} workers × {worker_thread_limit} threads)")

                except Exception as exc:
                    show_status("error", f"Échec génération grille: {exc}")
                    st.session_state.is_running = False
                    st.stop()

            # Choisir le mode fast_metrics pour les sweeps volumineux
            fast_metrics_env = os.getenv("BACKTEST_SWEEP_FAST_METRICS")
            if fast_metrics_env is not None:
                fast_metrics = fast_metrics_env.strip().lower() in ("1", "true", "yes", "on")
            else:
                try:
                    fast_metrics_threshold = int(os.getenv("BACKTEST_SWEEP_FAST_METRICS_THRESHOLD", "500"))
                except (TypeError, ValueError):
                    fast_metrics_threshold = 500
                fast_metrics = total_runs >= fast_metrics_threshold and not debug_enabled

            # Heuristique: éviter multiprocessing pour petites grilles (overhead > gain)
            try:
                min_parallel_runs = int(os.getenv("BACKTEST_SWEEP_MIN_PARALLEL", "200"))
            except (TypeError, ValueError):
                min_parallel_runs = 200
            min_parallel_runs = max(0, min_parallel_runs)
            run_parallel = (
                n_workers_effective > 1
                and total_runs > 1
                and (min_parallel_runs == 0 or total_runs >= min_parallel_runs)
            )
            if not run_parallel:
                n_workers_effective = 1

            st.caption(
                f"⚙️ Grille: {total_runs:,} combos | mode: "
                f"{'parallel' if run_parallel else 'séquentiel'} | "
                f"workers: {n_workers_effective} | threads: {worker_thread_limit} | "
                f"fast_metrics: {'on' if fast_metrics else 'off'}"
            )

            results_list = []
            param_combos_map = {}

            monitor = ProgressMonitor(total_runs=total_runs)
            monitor_placeholder = st.empty()

            sweep_monitor = SweepMonitor(
                total_combinations=total_runs,
                objectives=["total_pnl", "theoretical_pnl", "sharpe_ratio", "total_return_pct", "max_drawdown"],
                top_k=15,
            )
            sweep_monitor.start()
            sweep_placeholder = st.empty()

            logger = logging.getLogger(__name__)
            error_counts: Dict[str, int] = {}
            error_logged: set[str] = set()
            try:
                error_log_limit = int(os.environ.get("BACKTEST_SWEEP_ERROR_LOG_LIMIT", "3"))
            except (TypeError, ValueError):
                error_log_limit = 3

            st.markdown("### 📊 Progression en temps réel")
            render_progress_monitor(monitor, monitor_placeholder)


            def _normalize_param_combo(param_combo: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    k: float(v) if hasattr(v, "item") else v for k, v in param_combo.items()
                }

            def _params_to_str(param_combo: Dict[str, Any]) -> str:
                return str(_normalize_param_combo(param_combo))

            def run_single_backtest(param_combo: Dict[str, Any]):
                try:
                    result_i, msg_i = safe_run_backtest(
                        engine,
                        df,
                        strategy_key,
                        param_combo,
                        symbol,
                        timeframe,
                        silent_mode=True,
                        fast_metrics=fast_metrics,
                    )

                    params_str = _params_to_str(param_combo)

                    if result_i:
                        m = result_i.metrics
                        # Log des clés disponibles si debug activé
                        if debug_enabled and not m:
                            logger.warning("Metrics vides pour params=%s", params_str)
                        return {
                            "params": params_str,
                            "params_dict": param_combo,
                            "total_pnl": m.get("total_pnl", 0.0),
                            "theoretical_pnl": m.get("theoretical_pnl", 0.0),
                            "sharpe": m.get("sharpe_ratio", 0.0),
                            "max_dd": m.get("max_drawdown_pct", m.get("max_drawdown", 0.0)),
                            "win_rate": m.get("win_rate", 0.0),
                            "trades": m.get("total_trades", 0),
                            "profit_factor": m.get("profit_factor", 0.0),
                            "account_ruined": m.get("account_ruined", False),
                            "min_equity": m.get("min_equity", 0.0),
                            "consecutive_losses_max": m.get("consecutive_losses_max", 0),
                            "avg_win_loss_ratio": m.get("avg_win_loss_ratio", 0.0),
                            "robustness_score": m.get("robustness_score", 0.0),
                            "liquidation_total_pnl": m.get("liquidation_total_pnl", m.get("total_pnl", 0.0)),
                            "liquidation_total_return_pct": m.get("liquidation_total_return_pct", m.get("total_return_pct", 0.0)),
                            "liquidation_sharpe_ratio": m.get("liquidation_sharpe_ratio", m.get("sharpe_ratio", 0.0)),
                            "liquidation_max_drawdown_pct": m.get("liquidation_max_drawdown_pct", m.get("max_drawdown_pct", 0.0)),
                            "liquidation_triggered": m.get("liquidation_triggered", False),
                            "liquidation_time": m.get("liquidation_time"),
                            "period_days": period_days,
                        }
                    return {
                        "params": params_str,
                        "params_dict": param_combo,
                        "error": msg_i,
                    }
                except Exception as exc:
                    params_str = _params_to_str(param_combo)
                    # Log complet de l'erreur avec traceback
                    if debug_enabled:
                        logger.error("Backtest error params=%s: %s", params_str, traceback.format_exc())
                    return {
                        "params": params_str,
                        "params_dict": param_combo,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            def record_sweep_result(
                result: Dict[str, Any],
                fallback_params: Dict[str, Any],
            ) -> str:
                param_combo_result = result.get("params_dict") or fallback_params
                params_str = result.get("params") or _params_to_str(param_combo_result)
                result["params"] = params_str
                param_combos_map[params_str] = param_combo_result

                if "error" not in result:
                    metrics = {
                        "sharpe_ratio": result.get("sharpe", 0.0),
                        "total_pnl": result.get("total_pnl", 0.0),
                        "theoretical_pnl": result.get("theoretical_pnl", 0.0),
                        "total_return_pct": result.get("total_pnl", 0.0) / state.initial_capital * 100 if state.initial_capital else 0.0,
                        "max_drawdown": abs(result.get("max_dd", 0.0)),
                        "account_ruined": result.get("account_ruined", False),
                        "min_equity": result.get("min_equity", 0.0),
                        "win_rate": result.get("win_rate", 0.0),
                        "total_trades": result.get("trades", 0),
                        "profit_factor": result.get("profit_factor", 0.0),
                        "consecutive_losses_max": result.get("consecutive_losses_max", 0),
                        "avg_win_loss_ratio": result.get("avg_win_loss_ratio", 0.0),
                        "robustness_score": result.get("robustness_score", 0.0),
                        "period_days": result.get("period_days", period_days),
                    }
                    sweep_monitor.update(params=param_combo_result, metrics=metrics)
                else:
                    error_msg = str(result.get("error") or "Erreur inconnue")
                    error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                    if len(error_logged) < error_log_limit and error_msg not in error_logged:
                        logger.error("Sweep error sample: %s", error_msg)
                        error_logged.add(error_msg)
                    sweep_monitor.update(params=param_combo_result, metrics={}, error=True)

                result_clean = {k: v for k, v in result.items() if k != "params_dict"}
                results_list.append(result_clean)
                return params_str

            completed_params = set()
            completed = 0
            start_time = time.perf_counter()
            last_render_time = start_time

            def run_sequential_combos(combo_source, key_prefix: str) -> None:
                nonlocal completed, last_render_time
                for param_combo in combo_source:
                    params_str = _params_to_str(param_combo)
                    if params_str in completed_params:
                        continue

                    completed += 1
                    monitor.runs_completed = completed

                    result = run_single_backtest(param_combo)
                    params_str = record_sweep_result(result, param_combo)
                    completed_params.add(params_str)

                    current_time = time.perf_counter()
                    # 🚀 AFFICHAGE TEMPS RÉEL: Update fréquent pour voir progression live
                    # Update tous les 100 runs OU toutes les 2 secondes
                    # Avec graphiques désactivés, pas de surcharge WebSocket
                    if completed % 100 == 0 or current_time - last_render_time >= 2.0:
                        render_progress_monitor(monitor, monitor_placeholder)
                        # Réactiver graphiques avec throttling ultra lent + mode statique
                        with sweep_placeholder.container():
                            progress_pct = (completed / total_runs * 100) if total_runs > 0 else 0
                            # Calculer backtests/sec
                            elapsed = time.perf_counter() - start_time
                            bt_per_sec = completed / elapsed if elapsed > 0 else 0
                            st.info(f"⚡ {completed}/{total_runs} runs ({progress_pct:.1f}%) • 🚀 {bt_per_sec:.1f} bt/s")

                            # Afficher le meilleur PnL + métriques critiques
                            if hasattr(sweep_monitor, '_results') and sweep_monitor._results:
                                # Extraire métriques - optimisé avec un seul parcours
                                cumulative_pnl = 0.0
                                pnl_values = []
                                best_sharpe = float("-inf")
                                best_pf = 0.0
                                best_robustness = 0.0
                                best_run = None
                                period_days_sum = 0.0
                                period_days_count = 0

                                for r in sweep_monitor._results:
                                    pnl = r.metrics.get("total_pnl", 0.0)
                                    cumulative_pnl += pnl
                                    pnl_values.append(pnl)

                                    sharpe = r.metrics.get("sharpe_ratio", float("-inf"))
                                    pf = r.metrics.get("profit_factor", 0.0)
                                    rob = r.metrics.get("robustness_score", 0.0)

                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                    if pf > best_pf:
                                        best_pf = pf
                                    if rob > best_robustness:
                                        best_robustness = rob
                                        best_run = r

                                    # Moyenne period_days
                                    if 'period_days' in r.metrics:
                                        period_days_sum += r.metrics['period_days']
                                        period_days_count += 1

                                # Fallback sur meilleur PnL si aucun robustness valide
                                if best_run is None:
                                    best_run = max(sweep_monitor._results, key=lambda r: r.metrics.get("total_pnl", float("-inf")))

                                best_wl_ratio = best_run.metrics.get("avg_win_loss_ratio", 0.0)
                                best_consec_losses = best_run.metrics.get("consecutive_losses_max", 0)

                                # PnL moyen + meilleur + pire avec moyenne period_days
                                avg_period_days = period_days_sum / period_days_count if period_days_count > 0 else period_days
                                avg_pnl = cumulative_pnl / len(pnl_values) if pnl_values else 0.0
                                best_pnl = max(pnl_values) if pnl_values else 0.0
                                worst_pnl = min(pnl_values) if pnl_values else 0.0  # PnL le plus négatif
                                best_pnl_display = format_pnl_with_daily(
                                    best_pnl,
                                    avg_period_days,
                                    show_plus=True,
                                    escape_markdown=True,
                                )
                                avg_pnl_display = format_pnl_with_daily(
                                    avg_pnl,
                                    avg_period_days,
                                    show_plus=True,
                                    escape_markdown=True,
                                )
                                worst_pnl_display = format_pnl_with_daily(
                                    worst_pnl,
                                    avg_period_days,
                                    show_plus=True,
                                    escape_markdown=True,
                                )

                                # Affichage compact des métriques critiques avec worst PnL
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"💰 **Meilleur PnL**: **{best_pnl_display}**")
                                    st.caption(
                                        f"📊 PnL moyen: **{avg_pnl_display}**"
                                    )
                                    # Worst PnL avec indicateur de risque liquidation
                                    liquidation_icon = " ⚠️" if worst_pnl < -5000 else " ⚠️" if worst_pnl < 0 else ""
                                    worst_color = "red" if worst_pnl < 0 else "green"
                                    st.caption(
                                        f"📉 Pire PnL: :{worst_color}[**{worst_pnl_display}**]{liquidation_icon}"
                                    )
                                with col2:
                                    robustness_color = "green" if best_robustness > 2.0 else "orange" if best_robustness > 1.0 else "red"
                                    st.markdown(f"🎯 **Robustesse**: :{robustness_color}[**{best_robustness:.2f}**]")
                                    st.caption(
                                        f"Best Sharpe: **{best_sharpe:.2f}** | "
                                        f"Best PF: **{best_pf:.2f}**"
                                    )
                                    st.caption(f"📈 Sharpe × PF (idéal >3.0)")
                                with col3:
                                    wl_color = "green" if best_wl_ratio > 2.0 else "orange" if best_wl_ratio > 1.5 else "red"
                                    st.markdown(f"⚖️ **W/L Ratio**: :{wl_color}[**{best_wl_ratio:.2f}**]")
                                    st.caption(f"💔 Max pertes: **{best_consec_losses}** consécutives")

                            # Graphiques avec mode statique (pas d'interactivité = moins de données WebSocket)
                            render_sweep_progress(
                                sweep_monitor,
                                key=f"sweep_progress_seq_{completed}",
                                static_plots=True,  # Désactiver interactivité Plotly
                                show_top_results=True,  # Afficher meilleur/moyen PnL gaming-style
                                show_evolution=False  # Pas d'évolution pendant le sweep
                            )
                            st.caption(f"🔄 Rafraîchissement: tous les 100 runs ou 2s (temps réel)")
                        last_render_time = current_time
                        time.sleep(0.01)

            if run_parallel:
                from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, TimeoutError as FutureTimeoutError, wait
                try:
                    from concurrent.futures import BrokenProcessPool
                except ImportError:  # pragma: no cover - fallback for older runtimes
                    BrokenProcessPool = RuntimeError

                # Système de diagnostic
                from utils.sweep_diagnostics import SweepDiagnostics
                diag = SweepDiagnostics(run_id=f"grid_{strategy_key}")
                diag.log_pool_start(n_workers_effective, worker_thread_limit, total_runs)

                logger = logging.getLogger(__name__)
                stall_timeout_sec = float(os.getenv("BACKTEST_SWEEP_STALL_SEC", "60"))
                max_inflight = max(1, min(total_runs, n_workers_effective * 2))
                pending = {}
                failed_pending = []
                pool_failed = False
                pool_fail_reason = None
                pool_error: Exception | None = None
                last_completion_time = time.perf_counter()
                pickle_error_count = 0  # Compteur d'erreurs de pickling
                combo_counter = 0  # Compteur pour diagnostics
                start_time = time.perf_counter()  # Pour calcul bt/s

                def submit_next() -> bool:
                    nonlocal combo_counter
                    try:
                        param_combo = next(combo_iter)
                    except StopIteration:
                        return False
                    combo_counter += 1
                    diag.log_submit(combo_counter, param_combo)
                    # Soumettre UNIQUEMENT param_combo - le DataFrame est déjà dans le worker
                    # Cela évite la sérialisation pickle répétée du DataFrame (économie CPU + mémoire)
                    future = executor.submit(
                        _isolated_worker,
                        param_combo  # Seul paramètre : dict des paramètres de stratégie
                    )
                    pending[future] = param_combo
                    return True

                # Import de l'initializer optimisé qui charge le DataFrame une seule fois par worker
                from backtest.worker import init_worker_with_dataframe

                executor = ProcessPoolExecutor(
                    max_workers=n_workers_effective,
                    initializer=init_worker_with_dataframe,
                    initargs=(
                        df,  # DataFrame chargé UNE SEULE FOIS par worker
                        strategy_key,
                        symbol,
                        timeframe,
                        state.initial_capital,
                        debug_enabled,
                        worker_thread_limit,
                        fast_metrics,
                    ),
                )
                try:
                    for _ in range(max_inflight):
                        if not submit_next():
                            break

                    while pending:
                        done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        if not done:
                            if time.perf_counter() - last_completion_time >= stall_timeout_sec:
                                pool_failed = True
                                pool_fail_reason = "stall"
                                pool_error = TimeoutError(
                                    f"Aucune completion depuis {stall_timeout_sec:.0f}s"
                                )
                                diag.log_stall(stall_timeout_sec, list(pending.values()))
                                logger.error(
                                    "Sweep multiprocess bloque depuis %ss, bascule sequentielle.",
                                    int(stall_timeout_sec),
                                )
                                break
                            continue

                        for future in done:
                            param_combo = pending.pop(future)
                            future_start = time.perf_counter()
                            result = None
                            should_record = True

                            try:
                                # Timeout 300s pour éviter freeze si Windows interrupt (Task Manager, focus change, etc.)
                                result = future.result(timeout=300)
                                duration_ms = (time.perf_counter() - future_start) * 1000

                                # Log completion
                                combo_idx = combo_counter - len(pending) - len(failed_pending)
                                diag.log_completion(combo_idx, param_combo, result, duration_ms)

                                # Détecter erreur de pickling dans le résultat
                                if isinstance(result, dict) and result.get("error", ""):
                                    error_msg = str(result.get("error", ""))
                                    if "pickle" in error_msg.lower() or "not the same object" in error_msg:
                                        pickle_error_count += 1
                                        if pickle_error_count >= 10:
                                            pool_failed = True
                                            pool_fail_reason = "pickle"
                                            pool_error = RuntimeError(
                                                "Erreur de pickling détectée - Streamlit a rechargé le module. "
                                                "Relancez le sweep après le rechargement."
                                            )
                                            logger.error(
                                                "Erreur de pickling répétée (%d fois), arrêt du sweep.",
                                                pickle_error_count,
                                            )
                                            failed_pending.append(param_combo)
                                            should_record = False
                                            break

                            except BrokenProcessPool as exc:
                                combo_idx = combo_counter - len(pending) - len(failed_pending)
                                diag.log_pool_broken("BrokenProcessPool", exc)
                                pool_failed = True
                                pool_fail_reason = "broken"
                                pool_error = exc
                                failed_pending.append(param_combo)
                                should_record = False

                                break

                            except FutureTimeoutError:
                                # Worker timeout (>300s) - probablement bloqué par interruption Windows
                                combo_idx = combo_counter - len(pending) - len(failed_pending)
                                diag.log_timeout(combo_idx, param_combo, 300)
                                logger.warning("Worker timeout (>300s) combo: %s", param_combo)
                                result = {
                                    "params": _params_to_str(param_combo),
                                    "params_dict": param_combo,
                                    "error": "Worker timeout (>300s, probablement bloqué par interruption Windows)",
                                }
                                # should_record reste True - on enregistre le timeout comme erreur

                            except Exception as exc:
                                combo_idx = combo_counter - len(pending) - len(failed_pending)
                                diag.log_future_exception(combo_idx, param_combo, exc)
                                error_str = f"{type(exc).__name__}: {exc}"
                                # Détecter erreur de pickling dans l'exception
                                if "pickle" in error_str.lower() or "not the same object" in error_str:
                                    pickle_error_count += 1
                                    if pickle_error_count >= 10:
                                        pool_failed = True
                                        pool_fail_reason = "pickle"
                                        pool_error = RuntimeError(
                                            "Erreur de pickling - le module a été rechargé pendant le sweep."
                                        )
                                        failed_pending.append(param_combo)
                                        should_record = False
                                        break
                                result = {
                                    "params": _params_to_str(param_combo),
                                    "params_dict": param_combo,
                                    "error": error_str,
                                }
                                # should_record reste True - on enregistre l'erreur

                            # Enregistrer le résultat (sauf si break anticipé)
                            if should_record and result is not None:
                                completed += 1
                                monitor.runs_completed = completed
                                params_str = record_sweep_result(result, param_combo)
                                completed_params.add(params_str)
                                last_completion_time = time.perf_counter()

                            # ⚡ CRITIQUE: Soumettre la combinaison suivante UNE SEULE FOIS après traitement complet
                            # (sauf si pool_failed ou break - dans ce cas on sort de la boucle de toute façon)
                            if not pool_failed:
                                submit_next()

                            current_time = time.perf_counter()
                            # 🚀 AFFICHAGE TEMPS RÉEL: Update fréquent pour voir progression live
                            # Update tous les 100 runs OU toutes les 2 secondes
                            if completed % 100 == 0 or current_time - last_render_time >= 2.0:
                                render_progress_monitor(monitor, monitor_placeholder)
                                # Réactiver graphiques avec throttling ultra lent + mode statique
                                with sweep_placeholder.container():
                                    progress_pct = (completed / total_runs * 100) if total_runs > 0 else 0
                                    # Calculer backtests/sec
                                    elapsed = time.perf_counter() - start_time
                                    bt_per_sec = completed / elapsed if elapsed > 0 else 0
                                    st.info(f"⚡ {completed}/{total_runs} runs ({progress_pct:.1f}%) • 🚀 {bt_per_sec:.1f} bt/s")

                                    # Afficher le meilleur PnL + métriques critiques
                                    if hasattr(sweep_monitor, '_results') and sweep_monitor._results:
                                        # Extraire métriques - optimisé avec un seul parcours
                                        cumulative_pnl = 0.0
                                        pnl_values = []
                                        best_sharpe = float("-inf")
                                        best_pf = 0.0
                                        best_robustness = 0.0
                                        best_run = None
                                        period_days_sum = 0.0
                                        period_days_count = 0

                                        for r in sweep_monitor._results:
                                            pnl = r.metrics.get("total_pnl", 0.0)
                                            cumulative_pnl += pnl
                                            pnl_values.append(pnl)

                                            sharpe = r.metrics.get("sharpe_ratio", float("-inf"))
                                            pf = r.metrics.get("profit_factor", 0.0)
                                            rob = r.metrics.get("robustness_score", 0.0)

                                            if sharpe > best_sharpe:
                                                best_sharpe = sharpe
                                            if pf > best_pf:
                                                best_pf = pf
                                            if rob > best_robustness:
                                                best_robustness = rob
                                                best_run = r

                                            # Moyenne period_days
                                            if 'period_days' in r.metrics and r.metrics['period_days'] is not None:
                                                period_days_sum += r.metrics['period_days']
                                                period_days_count += 1

                                        # Fallback sur meilleur PnL si aucun robustness valide
                                        if best_run is None:
                                            best_run = max(sweep_monitor._results, key=lambda r: r.metrics.get("total_pnl", float("-inf")))

                                        best_wl_ratio = best_run.metrics.get("avg_win_loss_ratio", 0.0)
                                        best_consec_losses = best_run.metrics.get("consecutive_losses_max", 0)

                                        # PnL moyen + meilleur avec moyenne period_days
                                        avg_period_days = period_days_sum / period_days_count if period_days_count > 0 else period_days
                                        avg_pnl = cumulative_pnl / len(pnl_values) if pnl_values else 0.0
                                        best_pnl = max(pnl_values) if pnl_values else 0.0
                                        best_pnl_display = format_pnl_with_daily(
                                            best_pnl,
                                            avg_period_days,
                                            show_plus=True,
                                            escape_markdown=True,
                                        )
                                        avg_pnl_display = format_pnl_with_daily(
                                            avg_pnl,
                                            avg_period_days,
                                            show_plus=True,
                                            escape_markdown=True,
                                        )

                                        # Affichage compact des métriques critiques
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.markdown(f"💰 **Meilleur PnL**: **{best_pnl_display}**")
                                            st.caption(
                                                f"📊 PnL moyen: **{avg_pnl_display}** | "
                                                f"Best Sharpe: **{best_sharpe:.2f}** | "
                                                f"Best PF: **{best_pf:.2f}**"
                                            )
                                        with col2:
                                            robustness_color = "green" if best_robustness > 2.0 else "orange" if best_robustness > 1.0 else "red"
                                            st.markdown(f"🎯 **Robustesse**: :{robustness_color}[**{best_robustness:.2f}**]")
                                            st.caption(f"📈 Sharpe × PF (idéal >3.0)")
                                        with col3:
                                            wl_color = "green" if best_wl_ratio > 2.0 else "orange" if best_wl_ratio > 1.5 else "red"
                                            st.markdown(f"⚖️ **W/L Ratio**: :{wl_color}[**{best_wl_ratio:.2f}**]")
                                            st.caption(f"💔 Max pertes: **{best_consec_losses}** consécutives")

                                    # Graphiques avec mode statique
                                    render_sweep_progress(
                                        sweep_monitor,
                                        key=f"sweep_progress_multi_{completed}",
                                        static_plots=True,
                                        show_top_results=True,  # Afficher meilleur/moyen PnL gaming-style
                                        show_evolution=False  # Pas d'évolution pendant le sweep
                                    )
                                    st.caption(f"🔄 Rafraîchissement: tous les 100 runs ou 2s (temps réel)")
                                last_render_time = current_time
                                time.sleep(0.01)

                        if pool_failed:
                            diag.log_pool_broken(pool_fail_reason or "unknown", pool_error)
                            break
                finally:
                    diag.log_pool_shutdown(success=not pool_failed)
                    try:
                        executor.shutdown(
                            wait=not pool_failed,
                            cancel_futures=pool_failed,
                        )
                    except Exception:
                        logger.exception("Erreur shutdown ProcessPoolExecutor")

                if pool_failed:
                    with status_container:
                        if pool_fail_reason == "pickle":
                            show_status(
                                "error",
                                "⚠️ Erreur de pickling: le module a été rechargé par Streamlit pendant le sweep. "
                                "Relancez le sweep - il reprendra depuis les combinaisons non testées.",
                            )
                        else:
                            show_status(
                                "warning",
                                "Pool multiprocess interrompu, reprise en mode séquentiel.",
                            )
                        if pool_error:
                            st.caption(f"Détails: {pool_error}")

                    pending_combos = failed_pending + list(pending.values())
                    if pool_fail_reason == "stall" and pending_combos:
                        for param_combo in pending_combos:
                            completed += 1
                            monitor.runs_completed = completed
                            params_str = record_sweep_result(
                                {"params_dict": param_combo, "error": "worker_stall"},
                                param_combo,
                            )
                            completed_params.add(params_str)
                        pending_combos = []

                    diag.log_sequential_fallback(len(pending_combos))
                    fallback_iter = chain(pending_combos, combo_iter)
                    run_sequential_combos(fallback_iter, "sweep_fallback")
            else:
                run_sequential_combos(combo_iter, "sweep_sequential")

            render_progress_monitor(monitor, monitor_placeholder)
            sweep_placeholder.empty()
            with sweep_placeholder.container():
                render_sweep_progress(
                    sweep_monitor,
                    key="sweep_final",
                    show_top_results=True,
                    show_evolution=True,
                )

            st.markdown("---")
            st.markdown("### 🎯 Résumé de l'Optimisation")
            render_sweep_summary(sweep_monitor, key="sweep_summary")

            # Finalize diagnostics
            diag.log_final_summary()
            st.caption(f"📋 Logs diagnostiques: `{diag.log_file}`")

            monitor_placeholder.empty()
            sweep_placeholder.empty()

            with status_container:
                show_status("success", f"Optimisation: {len(results_list)} tests")

            results_df = pd.DataFrame(results_list)

            if "trades" in results_df.columns:
                logger = logging.getLogger(__name__)
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

            error_items = []
            if error_counts:
                total_errors = sum(error_counts.values())
                with st.expander("❌ Erreurs (extraits)", expanded=True):
                    st.caption(
                        f"{total_errors} erreurs detectees. "
                        "Consultez le terminal pour les premiers messages."
                    )
                    error_items = sorted(
                        error_counts.items(), key=lambda item: item[1], reverse=True
                    )
                    error_df = pd.DataFrame(
                        [
                            {"error": msg, "count": count}
                            for msg, count in error_items[:10]
                        ]
                    )
                    st.dataframe(error_df, width="stretch")

            error_column = results_df.get("error")
            if error_column is not None:
                valid_results = results_df[error_column.isna()]
            else:
                valid_results = results_df

            if not valid_results.empty:
                valid_results = valid_results.sort_values("sharpe", ascending=False)

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
                    fast_metrics=False,
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
                # Afficher diagnostic détaillé
                st.markdown("### 🔍 Diagnostic")
                st.warning(
                    f"Sur {len(results_list)} combinaisons évaluées, "
                    f"toutes ont échoué."
                )
                if error_items:
                    top_error, top_count = error_items[0]
                    st.error(
                        f"**Erreur principale** ({top_count} occurrences sur {sum(error_counts.values())} erreurs):"
                    )
                    st.code(top_error, language="text")
                elif results_list:
                    # Extraire les erreurs du DataFrame si error_counts vide
                    errors_in_results = [
                        r.get("error") for r in results_list if r.get("error")
                    ]
                    if errors_in_results:
                        st.error(f"**Première erreur détectée:**")
                        st.code(errors_in_results[0], language="text")
                        if len(errors_in_results) > 1:
                            st.caption(f"+ {len(errors_in_results)-1} autres erreurs similaires")
                    else:
                        st.info(
                            "Aucune erreur explicite, mais les résultats sont invalides. "
                            "Vérifiez que les données OHLCV sont chargées et valides."
                        )
                st.session_state.is_running = False
                st.stop()

        elif optimization_mode == "🤖 Optimisation LLM":
            handle_llm_optimization(
                state=state,
                df=df,
                engine=engine,
                status_container=status_container,
            )

        else:
            show_status("error", f"Mode non reconnu: {optimization_mode}")
            st.session_state.is_running = False
            st.stop()

    st.session_state.is_running = False
