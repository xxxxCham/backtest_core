"""
Module-ID: ui.sidebar

Purpose: Gère la configuration et les contrôles de la sidebar pour la sélection de stratégies et paramètres.

Role in pipeline: configuration / inputs

Key components: render_sidebar, gestion des paramètres

Inputs: Données disponibles, stratégies

Outputs: SidebarState configuré

Dependencies: ui.context, ui.constants

Conventions: Paramètres validés selon contraintes

Read-if: Configuration de l'interface utilisateur

Skip-if: Logique backend pure
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ui.constants import (
    MODE_BUTTON_CSS,
    MODE_OPTIONS,
    PARAM_CONSTRAINTS,
    build_strategy_options,
    get_strategy_description,
    get_strategy_ui_indicators,
)
from ui.context import (
    KNOWN_MODELS,
    LLM_AVAILABLE,
    LLM_IMPORT_ERROR,
    RECOMMENDED_FOR_STRATEGY,
    LLMConfig,
    LLMProvider,
    ModelCategory,
    compute_search_space_stats,
    discover_available_data,
    ensure_ollama_running,
    get_available_models_for_ui,
    get_global_model_config,
    get_model_info,
    get_storage,
    get_strategy,
    get_strategy_info,
    is_ollama_available,
    list_available_models,
    list_strategies,
    list_strategy_versions,
    load_strategy_version,
    resolve_latest_version,
    set_global_model_config,
)
from ui.helpers import (
    _data_cache_key,
    _find_saved_run_meta,
    _parse_run_timestamp,
    apply_versioned_preset,
    create_param_range_selector,
    load_selected_data,
    render_saved_runs_panel,
    validate_param,
)
from ui.state import SidebarState
from utils.observability import is_debug_enabled, set_log_level


def _is_valid_timeframe_format(tf: str) -> bool:
    """Valide qu'un timeframe est dans un format correct."""
    if not tf or len(tf) < 2:
        return False
    unit = tf[-1]
    if unit not in ('m', 'h', 'd', 'w', 'M'):
        return False
    try:
        amount = int(tf[:-1])
        return amount > 0
    except ValueError:
        return False


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_signature_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_signature_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize_signature_value(v) for v in value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_padded_date_range(
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    pad_days: int = 1,
) -> Tuple[Optional[object], Optional[object]]:
    if start_ts is None or end_ts is None:
        return None, None
    start_date = start_ts.date()
    end_date = end_ts.date()
    if pad_days <= 0:
        return start_date, end_date
    padded_start = (start_ts + pd.Timedelta(days=pad_days)).date()
    padded_end = (end_ts - pd.Timedelta(days=pad_days)).date()
    if padded_start < padded_end:
        return padded_start, padded_end
    return start_date, end_date


def _extract_llm_signature(llm_config: Optional[LLMConfig]) -> Optional[Dict[str, Any]]:
    if llm_config is None:
        return None
    provider = getattr(llm_config.provider, "value", str(llm_config.provider))
    return {
        "provider": provider,
        "model": getattr(llm_config, "model", None),
        "ollama_host": getattr(llm_config, "ollama_host", None),
        "openai_base_url": getattr(llm_config, "openai_base_url", None),
        "openai_key_set": bool(getattr(llm_config, "openai_api_key", None)),
        "temperature": getattr(llm_config, "temperature", None),
        "max_tokens": getattr(llm_config, "max_tokens", None),
        "top_p": getattr(llm_config, "top_p", None),
        "timeout_seconds": getattr(llm_config, "timeout_seconds", None),
        "max_retries": getattr(llm_config, "max_retries", None),
        "retry_delay_seconds": getattr(llm_config, "retry_delay_seconds", None),
    }


def _extract_role_model_signature(role_model_config: Any) -> Optional[Dict[str, Any]]:
    if role_model_config is None:
        return None

    def _role_payload(role: Any) -> Dict[str, Any]:
        return {
            "models": list(getattr(role, "models", []) or []),
            "allow_heavy_after_iteration": getattr(role, "allow_heavy_after_iteration", None),
        }

    return {
        "analyst": _role_payload(getattr(role_model_config, "analyst", None)),
        "strategist": _role_payload(getattr(role_model_config, "strategist", None)),
        "critic": _role_payload(getattr(role_model_config, "critic", None)),
        "validator": _role_payload(getattr(role_model_config, "validator", None)),
    }


def _build_config_signature(state: SidebarState) -> str:
    """Construit une signature stable de la configuration appliquée."""
    payload = {
        "debug_enabled": state.debug_enabled,
        "symbol": state.symbol,
        "timeframe": state.timeframe,
        "use_date_filter": state.use_date_filter,
        "start_date": state.start_date,
        "end_date": state.end_date,
        "symbols": sorted(state.symbols or []),
        "timeframes": sorted(state.timeframes or []),
        "strategy_key": state.strategy_key,
        "strategy_keys": sorted(state.strategy_keys or []),
        "params": state.params,
        "param_ranges": state.param_ranges,
        "all_params": state.all_params,
        "all_param_ranges": state.all_param_ranges,
        "active_indicators": sorted(state.active_indicators or []),
        "optimization_mode": state.optimization_mode,
        "max_combos": state.max_combos,
        "n_workers": state.n_workers,
        "use_optuna": state.use_optuna,
        "optuna_n_trials": state.optuna_n_trials,
        "optuna_sampler": state.optuna_sampler,
        "optuna_pruning": state.optuna_pruning,
        "optuna_metric": state.optuna_metric,
        "optuna_early_stop": state.optuna_early_stop,
        "llm": _extract_llm_signature(state.llm_config),
        "llm_model": state.llm_model,
        "llm_use_multi_agent": state.llm_use_multi_agent,
        "role_model_config": _extract_role_model_signature(state.role_model_config),
        "llm_max_iterations": state.llm_max_iterations,
        "llm_use_walk_forward": state.llm_use_walk_forward,
        "llm_unload_during_backtest": state.llm_unload_during_backtest,
        "llm_compare_enabled": state.llm_compare_enabled,
        "llm_compare_auto_run": state.llm_compare_auto_run,
        "llm_compare_strategies": sorted(state.llm_compare_strategies or []),
        "llm_compare_tokens": sorted(state.llm_compare_tokens or []),
        "llm_compare_timeframes": sorted(state.llm_compare_timeframes or []),
        "llm_compare_metric": state.llm_compare_metric,
        "llm_compare_aggregate": state.llm_compare_aggregate,
        "llm_compare_max_runs": state.llm_compare_max_runs,
        "llm_compare_use_preset": state.llm_compare_use_preset,
        "llm_compare_generate_report": state.llm_compare_generate_report,
        "initial_capital": state.initial_capital,
        "leverage": state.leverage,
        "leverage_enabled": state.leverage_enabled,
        "disabled_params": sorted(state.disabled_params or []),
    }

    normalized = _normalize_signature_value(payload)
    return json.dumps(normalized, sort_keys=True, default=str)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _apply_config_guard(draft_state: SidebarState) -> SidebarState:
    draft_signature = _build_config_signature(draft_state)
    applied_signature = st.session_state.get("applied_config_signature")
    applied_state = st.session_state.get("applied_sidebar_state")

    if applied_signature is None or applied_state is None:
        st.session_state["applied_config_signature"] = draft_signature
        st.session_state["applied_sidebar_state"] = draft_state
        applied_state = draft_state
        pending = False
    else:
        pending = draft_signature != applied_signature

    st.session_state["config_pending_changes"] = pending
    st.session_state["draft_config_signature"] = draft_signature

    return applied_state


def render_sidebar() -> SidebarState:
    st.sidebar.header("⚙️ Configuration")

    with st.sidebar.expander("🔧 Debug", expanded=False):
        debug_enabled = st.checkbox(
            "Mode DEBUG",
            value=is_debug_enabled(),
            key="debug_toggle",
        )
        if debug_enabled:
            set_log_level("DEBUG")
            st.caption("🟢 Logs détaillés activés")
        else:
            set_log_level("INFO")

    st.sidebar.subheader("📊 Données")

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

        # Nettoyer les valeurs de session invalides (bug fix 23/01/2026)
        if "symbol_select" in st.session_state:
            if st.session_state["symbol_select"] not in available_tokens:
                del st.session_state["symbol_select"]

        if "timeframe_select" in st.session_state:
            if not _is_valid_timeframe_format(st.session_state["timeframe_select"]) or \
               st.session_state["timeframe_select"] not in available_timeframes:
                del st.session_state["timeframe_select"]

    except Exception as exc:
        available_tokens = ["BTCUSDC", "ETHUSDC"]
        available_timeframes = ["1h", "4h", "1d"]
        data_status.error(f"Erreur scan: {exc}")

    pending_meta = None
    pending_run_id = st.session_state.get("pending_run_load_id")
    if pending_run_id:
        try:
            storage = get_storage()
            pending_meta = _find_saved_run_meta(storage, pending_run_id)
        except Exception as exc:
            st.session_state["saved_runs_status"] = f"Pending load failed: {exc}"
            pending_meta = None

    if pending_meta is not None:
        # Valider que symbol et timeframe sont valides avant de les ajouter
        if pending_meta.symbol and pending_meta.symbol not in available_tokens:
            # Vérifier que le symbol est valide (lettres et chiffres seulement)
            if pending_meta.symbol.replace("_", "").replace("-", "").isalnum():
                available_tokens = [pending_meta.symbol] + available_tokens

        if pending_meta.timeframe and pending_meta.timeframe not in available_timeframes:
            # Valider format timeframe (ex: 1m, 5m, 1h, 4h, 1d)
            if _is_valid_timeframe_format(pending_meta.timeframe):
                available_timeframes = [pending_meta.timeframe] + available_timeframes

        if pending_meta.symbol:
            st.session_state["symbol_select"] = pending_meta.symbol
        if pending_meta.timeframe:
            st.session_state["timeframe_select"] = pending_meta.timeframe
        # Activer le filtre de dates seulement si des dates spécifiques sont définies
        start_ts = _parse_run_timestamp(pending_meta.period_start)
        end_ts = _parse_run_timestamp(pending_meta.period_end)
        if start_ts is not None and end_ts is not None:
            st.session_state["use_date_filter"] = True
            # Initialiser seulement si pas déjà défini (évite conflit avec widget)
            if "start_date" not in st.session_state:
                st.session_state["start_date"] = start_ts.date()
            if "end_date" not in st.session_state:
                st.session_state["end_date"] = end_ts.date()

    # === NETTOYAGE SESSION STATE ===
    # Nettoyer les clés de session obsolètes ou invalides
    # IMPORTANT: Ne supprimer QUE les tokens/timeframes vraiment invalides
    # Ne PAS réinitialiser si certains sont encore valides
    session_keys_to_clean = [
        "symbols_select", "timeframes_select", "symbol_select", "timeframe_select"
    ]
    for key in session_keys_to_clean:
        if key in st.session_state:
            if "symbol" in key:
                if isinstance(st.session_state[key], list):
                    # Multi-select : garder uniquement les valeurs valides
                    valid_symbols = [s for s in st.session_state[key] if s in available_tokens]
                    # Ne mettre à jour QUE si des symboles invalides ont été retirés
                    if valid_symbols and valid_symbols != st.session_state[key]:
                        st.session_state[key] = valid_symbols
                    elif not valid_symbols:
                        # Aucun symbole valide - réinitialiser
                        st.session_state[key] = available_tokens[:1] if available_tokens else []
                elif st.session_state[key] not in available_tokens:
                    del st.session_state[key]
            elif "timeframe" in key:
                if isinstance(st.session_state[key], list):
                    # Multi-select : garder uniquement les valeurs valides
                    valid_timeframes = [tf for tf in st.session_state[key] if tf in available_timeframes]
                    # Ne mettre à jour QUE si des timeframes invalides ont été retirés
                    if valid_timeframes and valid_timeframes != st.session_state[key]:
                        st.session_state[key] = valid_timeframes
                    elif not valid_timeframes:
                        # Aucun timeframe valide - réinitialiser
                        st.session_state[key] = available_timeframes[:1] if available_timeframes else []
                elif st.session_state[key] not in available_timeframes:
                    del st.session_state[key]

    # === MULTI-SÉLECTION TOKENS (multiselect) ===
    # Tokens à potentiel (base de comparaison méticuleuse)
    POTENTIAL_TOKENS = [
        "BTCUSDC",    # Bitcoin - Référence marché
        "ETHUSDC",    # Ethereum - Leader DeFi
        "BNBUSDC",    # Binance Coin - Plateforme CEX
        "SOLUSDC",    # Solana - Haute vitesse
        "AVAXUSDC",   # Avalanche - DeFi concurrente
        "LINKUSDC",   # Chainlink - Oracle leader
        "ADAUSDC",    # Cardano - Approche académique
        "DOTUSDC",    # Polkadot - Interopérabilité
        "ATOMUSDC",   # Cosmos - Hub inter-chaînes
    ]

    default_symbols = ["BTCUSDC"] if "BTCUSDC" in available_tokens else available_tokens[:1]

    # Appliquer la sélection des tokens potentiels avant la création du widget
    if st.session_state.get("_apply_potential_tokens", False):
        valid_potential = [t for t in POTENTIAL_TOKENS if t in available_tokens]
        current_symbols = st.session_state.get("symbols_select", default_symbols)
        merged_symbols = list(current_symbols)
        for token in valid_potential:
            if token not in merged_symbols:
                merged_symbols.append(token)
        st.session_state["symbols_select"] = merged_symbols or default_symbols
        del st.session_state["_apply_potential_tokens"]

    # IMPORTANT: S'assurer que symbols_select n'est jamais vide dans session_state
    # Si vide ou absent, initialiser avec default_symbols
    if "symbols_select" not in st.session_state or not st.session_state.get("symbols_select"):
        st.session_state["symbols_select"] = default_symbols

    # Layout: multiselect + bouton côte à côte
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        # Ne PAS utiliser default= car on utilise key= avec session_state pré-initialisé
        symbols = st.multiselect(
            label="Symbole(s)",
            options=available_tokens,
            key="symbols_select",
            help="Sélectionnez un ou plusieurs tokens à analyser",
        )
    with col2:
        st.write("")  # Espacement pour aligner avec le multiselect
        if st.button("🎯", key="select_potential_tokens", help="Sélectionner tokens à potentiel"):
            st.session_state["_apply_potential_tokens"] = True
            st.rerun()

    # Fallback si aucune sélection (double sécurité)
    if not symbols:
        symbols = default_symbols
        st.sidebar.warning("⚠️ Au moins un symbole requis. BTCUSDC sélectionné par défaut.")
    symbol = symbols[0]  # Compatibilité rétro

    # === MULTI-SÉLECTION TIMEFRAMES (multiselect) ===
    default_timeframes = ["30m"] if "30m" in available_timeframes else available_timeframes[:1]
    # IMPORTANT: S'assurer que timeframes_select n'est jamais vide
    if "timeframes_select" not in st.session_state or not st.session_state.get("timeframes_select"):
        st.session_state["timeframes_select"] = default_timeframes

    # Ne PAS utiliser default= car on utilise key= avec session_state pré-initialisé
    timeframes = st.sidebar.multiselect(
        "Timeframe(s)",
        available_timeframes,
        key="timeframes_select",
        help="Sélectionnez un ou plusieurs timeframes",
    )
    # Fallback si aucune sélection (double sécurité)
    if not timeframes:
        timeframes = default_timeframes
        st.sidebar.warning("⚠️ Au moins un timeframe requis. 30m sélectionné par défaut.")
    timeframe = timeframes[0]  # Compatibilité rétro

    # Info multi-sweep si plusieurs sélections (tokens/timeframes uniquement à ce stade)
    if len(symbols) > 1 or len(timeframes) > 1:
        total_combos = len(symbols) * len(timeframes)
        st.sidebar.info(f"🔄 Mode multi-sweep: {len(symbols)} token(s) × {len(timeframes)} TF(s) = {total_combos} combinaison(s)")

    # Analyse des données disponibles pour validation (toujours nécessaire)
    from data.config import scan_data_availability
    availability_result = scan_data_availability(symbols, timeframes)

    use_date_filter = st.sidebar.checkbox(
        "Filtrer par dates",
        value=False,
        help="Désactivé = utilise toutes les données disponibles (recommandé)",
        key="use_date_filter",
    )
    if use_date_filter:
        # === ANALYSE PAR CATÉGORIE DE TIMEFRAME ===
        from data.config import analyze_by_timeframe

        # Analyse par timeframe (plage commune par TF)
        timeframe_analysis = analyze_by_timeframe(symbols, timeframes)

        # Interface de sélection par timeframe
        with st.sidebar.expander("🎯 **Analyse par Timeframe**", expanded=True):
            if len(timeframes) > 1:
                analysis_mode = st.radio(
                    "Mode d'analyse",
                    ["Période harmonisée", "Périodes indépendantes par timeframe"],
                    help="Harmonisée = même période pour tous. Indépendantes = période optimale par timeframe",
                )
            else:
                analysis_mode = "Période harmonisée"  # Auto si un seul timeframe

            st.caption(
                "Harmonisée = une seule période commune (comparaisons strictes). "
                "Indépendantes = meilleure période par TF (comparaisons plus souples)."
            )

            available_start = None
            available_end = None
            default_start = None
            default_end = None

            if analysis_mode == "Période harmonisée":
                if availability_result.has_common_range:
                    common_start = availability_result.common_start
                    common_end = availability_result.common_end
                    duration = (common_end - common_start).days

                    st.success(f"✅ **Période harmonisée**: {common_start.strftime('%d/%m/%Y')} → {common_end.strftime('%d/%m/%Y')} ({duration}j)")
                    st.caption(
                        f"💡 Plage commune stricte (max début, min fin) sur "
                        f"{len(symbols)} token(s) × {len(timeframes)} TF(s)"
                    )

                    available_start = common_start.date()
                    available_end = common_end.date()
                    default_start, default_end = _get_padded_date_range(common_start, common_end)
                else:
                    st.warning("⚠️ Impossible de trouver une période commune (intersection vide)")
                    default_start = pd.Timestamp("2023-01-01").date()
                    default_end = pd.Timestamp.now().date()
                    available_start = default_start
                    available_end = default_end

            else:
                st.info("📊 **Périodes optimales par timeframe**:")

                best_timeframe = None
                best_score = 0.0
                best_period_ref = None

                for tf, data in timeframe_analysis.items():
                    st.write(f"**{tf}**")

                    if data['optimal_periods']:
                        best_period = data['optimal_periods'][0]
                        start_fr = best_period.start_date.strftime("%d/%m/%Y")
                        end_fr = best_period.end_date.strftime("%d/%m/%Y")
                        duration = (best_period.end_date - best_period.start_date).days

                        st.write(f"- 🎯 {start_fr} → {end_fr} ({duration}j)")
                        st.caption(
                            f"  Score: {best_period.completeness_score:.0f}%, "
                            f"Gap toléré: {data['gap_tolerance']:.0f}%"
                        )

                        for recommendation in data['recommendations']:
                            st.caption(f"  {recommendation}")

                        combined_score = best_period.completeness_score * best_period.avg_data_density
                        if combined_score > best_score:
                            best_score = combined_score
                            best_timeframe = tf
                            best_period_ref = best_period
                    else:
                        st.write("- ❌ Aucune période optimale trouvée")

                if best_timeframe and best_period_ref:
                    available_start = best_period_ref.start_date.date()
                    available_end = best_period_ref.end_date.date()
                    default_start, default_end = _get_padded_date_range(
                        best_period_ref.start_date,
                        best_period_ref.end_date,
                    )
                    st.success(f"🏆 **Défaut basé sur {best_timeframe}** (meilleur score: {best_score:.1f})")
                else:
                    st.warning("⚠️ Aucune période optimale trouvée pour les timeframes sélectionnés")
                    default_start = pd.Timestamp("2023-01-01").date()
                    default_end = pd.Timestamp.now().date()
                    available_start = default_start
                    available_end = default_end

            st.markdown("---")
            st.caption("📅 **Période d'analyse** (format: DD/MM/YYYY)")

            # Auto-aligner les dates sur la plage commune si hors limites.
            if default_start and default_end and available_start and available_end:
                selection_key = (
                    tuple(sorted(symbols)),
                    tuple(sorted(timeframes)),
                    analysis_mode,
                )
                if st.session_state.get("_date_range_selection_key") != selection_key:
                    st.session_state["start_date"] = default_start
                    st.session_state["end_date"] = default_end
                    st.session_state["_date_range_selection_key"] = selection_key

                start_state = st.session_state.get("start_date")
                end_state = st.session_state.get("end_date")
                if start_state and (start_state < available_start or start_state > available_end):
                    st.session_state["start_date"] = default_start
                if end_state and (end_state < available_start or end_state > available_end):
                    st.session_state["end_date"] = default_end

                if st.session_state.get("start_date") and st.session_state.get("end_date"):
                    if st.session_state["start_date"] >= st.session_state["end_date"]:
                        st.session_state["start_date"] = default_start
                        st.session_state["end_date"] = default_end

            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input(
                    "Date début 📅",
                    key="start_date",
                    format="DD/MM/YYYY",
                    help="Date de début de la période d'analyse"
                )
            with col2:
                end_date = st.date_input(
                    "Date fin 📅",
                    key="end_date",
                    format="DD/MM/YYYY",
                    help="Date de fin de la période d'analyse"
                )

            # Validation que start_date < end_date
            if start_date and end_date and start_date >= end_date:
                st.sidebar.error("⚠️ La date de début doit être antérieure à la date de fin")

            # Affichage de la durée sélectionnée
            if start_date and end_date and start_date < end_date:
                selected_days = (end_date - start_date).days
                st.sidebar.caption(f"📊 Durée sélectionnée: **{selected_days} jours**")

            # Validation de la période par rapport à la plage commune
            if availability_result.has_common_range and start_date and end_date:
                common_start = availability_result.common_start
                common_end = availability_result.common_end
                common_start_date = common_start.date()
                common_end_date = common_end.date()

                if analysis_mode == "Période harmonisée":
                    if end_date < common_start_date:
                        st.sidebar.error(
                            f"⚠️ Période demandée ({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}) est AVANT "
                            f"la plage commune ({common_start_date.strftime('%d/%m/%Y')})"
                        )
                    elif start_date > common_end_date:
                        st.sidebar.error(
                            f"⚠️ Période demandée ({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}) est APRÈS "
                            f"la plage commune ({common_end_date.strftime('%d/%m/%Y')})"
                        )
                    elif start_date < common_start_date:
                        st.sidebar.warning(
                            f"⚠️ Début demandé ({start_date.strftime('%d/%m/%Y')}) est AVANT la plage commune. "
                            f"Données réelles à partir de **{common_start_date.strftime('%d/%m/%Y')}**"
                        )
                    elif end_date > common_end_date:
                        st.sidebar.warning(
                            f"⚠️ Fin demandée ({end_date.strftime('%d/%m/%Y')}) est APRÈS la plage commune. "
                            f"Données réelles jusqu'à **{common_end_date.strftime('%d/%m/%Y')}**"
                        )
                else:
                    if start_date < common_start_date or end_date > common_end_date:
                        st.sidebar.info(
                            f"ℹ️ Plage commune globale: {common_start_date.strftime('%d/%m/%Y')} → {common_end_date.strftime('%d/%m/%Y')}. "
                            "En mode indépendant, certaines combinaisons peuvent être tronquées."
                        )

            st.markdown("---")
            with st.sidebar.expander("🔍 Analyse détaillée des données", expanded=False):
                if availability_result.rows:
                    df_analysis = pd.DataFrame(availability_result.rows)
                    st.dataframe(
                        df_analysis,
                        width="stretch",
                        column_config={
                            "Token": st.column_config.TextColumn("Token", width="small"),
                            "TF": st.column_config.TextColumn("TF", width="small"),
                            "Début": st.column_config.TextColumn("Début", width="medium"),
                            "Fin": st.column_config.TextColumn("Fin", width="medium"),
                            "Jours": st.column_config.NumberColumn("Jours", width="small"),
                            "Plage commune %": st.column_config.NumberColumn("Plage commune %", format="%.1f%%", width="small"),
                            "Couverture %": st.column_config.NumberColumn("Couverture %", format="%.1f%%", width="small"),
                            "Manquant %": st.column_config.NumberColumn("Manquant %", format="%.1f%%", width="small"),
                            "Jours manquants": st.column_config.NumberColumn("Jours manquants", format="%.1f", width="small"),
                            "Status": st.column_config.TextColumn("Status", width="small"),
                            "Détails": st.column_config.TextColumn("Détails", width="large")
                        }
                    )

                    total_combos = len(df_analysis)
                    complete_combos = len(df_analysis[df_analysis["Status"] == "✅"])
                    incomplete_combos = len(df_analysis[df_analysis["Status"] == "⚠️"])
                    missing_combos = len(df_analysis[df_analysis["Status"] == "❌"])

                    st.markdown("**Résumé qualité des données (gaps)**")
                    st.caption(
                        "✅ = couverture correcte (<10% de gaps) • ⚠️ = gaps significatifs • ❌ = fichier manquant."
                    )
                    st.markdown(
                        f"- ✅ Complètes : {complete_combos}/{total_combos}\n"
                        f"- ⚠️ Incomplètes : {incomplete_combos}/{total_combos}\n"
                        f"- ❌ Manquantes : {missing_combos}/{total_combos}"
                    )

                    if hasattr(availability_result, 'optimal_periods') and availability_result.optimal_periods:
                        st.markdown(
                            "💡 **Conseil :** Les périodes optimales ci-dessus évitent automatiquement les zones avec trop de données manquantes."
                        )
    else:
        start_date = None
        end_date = None

    current_data_key = _data_cache_key(symbol, timeframe, start_date, end_date)
    if st.session_state.get("ohlcv_cache_key") != current_data_key:
        st.session_state["ohlcv_cache_key"] = current_data_key
        st.session_state["ohlcv_df"] = None
        # FIX 04/01/2026: NE PAS effacer les résultats quand les données changent
        # Les résultats d'un backtest/grid peuvent être visualisés indépendamment
        # des données OHLCV actuellement chargées. Effacer les résultats causait
        # la perte de tous les résultats après un grid search lors du prochain rerun.
        # st.session_state["last_run_result"] = None
        # st.session_state["last_winner_params"] = None
        # st.session_state["last_winner_metrics"] = None
        # st.session_state["last_winner_origin"] = None
        # st.session_state["last_winner_meta"] = None

    pending_run_id = st.session_state.get("pending_run_load_id")
    if pending_run_id:
        try:
            storage = get_storage()
            result = storage.load_result(pending_run_id)
            st.session_state["last_run_result"] = result
            st.session_state["last_winner_params"] = result.meta.get("params", {})
            st.session_state["last_winner_metrics"] = result.metrics
            st.session_state["last_winner_origin"] = "storage"
            st.session_state["last_winner_meta"] = result.meta
            if st.session_state.get("pending_run_load_data", True):
                df_loaded, msg = load_selected_data(
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                )
                if df_loaded is None:
                    st.session_state["saved_runs_status"] = f"Data load failed: {msg}"
                else:
                    st.session_state["saved_runs_status"] = f"Run loaded with data: {msg}"
            else:
                st.session_state["saved_runs_status"] = f"Run loaded: {pending_run_id}"
        except Exception as exc:
            st.session_state["saved_runs_status"] = f"Load failed: {exc}"
        st.session_state.pop("pending_run_load_id", None)
        st.session_state.pop("pending_run_load_data", None)

    if st.session_state.get("ohlcv_df") is None:
        st.sidebar.info("Donnees non chargees.")
    else:
        cached_msg = st.session_state.get("ohlcv_status_msg", "")
        if cached_msg:
            st.sidebar.caption(f"Cache: {cached_msg}")

    st.sidebar.subheader("🎯 Stratégie")

    available_strategies = list_strategies()
    strategy_options = build_strategy_options(available_strategies)

    # === MULTI-SÉLECTION STRATÉGIES (multiselect) ===
    default_strategies = list(strategy_options.keys())[:1] if strategy_options else []
    strategy_names = st.sidebar.multiselect(
        "Stratégie(s)",
        list(strategy_options.keys()),
        default=default_strategies,
        key="strategies_select",
        help="Sélectionnez une ou plusieurs stratégies",
    )

    # Fallback si aucune sélection
    if not strategy_names:
        strategy_names = default_strategies
        st.sidebar.warning("⚠️ Au moins une stratégie requise. Première stratégie sélectionnée par défaut.")

    # Info multi-stratégies si plusieurs sélections
    if len(strategy_names) > 1:
        st.sidebar.info(
            f"📋 **{len(strategy_names)} stratégies sélectionnées**\n\n"
            f"Paramètres configurables pour: **{strategy_names[0]}**\n\n"
            f"Autres stratégies utiliseront leurs paramètres par défaut."
        )

    # Compatibilité rétro: première stratégie pour l'affichage des paramètres
    strategy_name = strategy_names[0]
    strategy_key = strategy_options[strategy_name]

    # Message multi-sweep global (stratégies + tokens + timeframes)
    if len(strategy_names) > 1 or len(symbols) > 1 or len(timeframes) > 1:
        total_combos = len(strategy_names) * len(symbols) * len(timeframes)
        parts = []
        if len(strategy_names) > 1:
            parts.append(f"{len(strategy_names)} stratégie(s)")
        if len(symbols) > 1:
            parts.append(f"{len(symbols)} token(s)")
        if len(timeframes) > 1:
            parts.append(f"{len(timeframes)} TF(s)")

        if len(parts) > 1:  # Seulement si au moins 2 dimensions multiples
            st.sidebar.success(f"🔄 **Multi-sweep total**: {' × '.join(parts)} = **{total_combos} backtests**")

    st.sidebar.caption(get_strategy_description(strategy_key))

    strategy_info = None
    try:
        strategy_info = get_strategy_info(strategy_key)

        if strategy_info.required_indicators:
            indicators_list = ", ".join(
                [f"**{ind.upper()}**" for ind in strategy_info.required_indicators]
            )
            st.sidebar.info(f"📊 Indicateurs requis: {indicators_list}")
        else:
            st.sidebar.info("📊 Indicateurs: Calculés internement")

        if strategy_info.internal_indicators:
            internal_list = ", ".join(
                [f"{ind.upper()}" for ind in strategy_info.internal_indicators]
            )
            st.sidebar.caption(f"_Calculés: {internal_list}_")

    except KeyError:
        st.sidebar.warning(f"⚠️ Indicateurs non définis pour '{strategy_key}'")

    st.sidebar.subheader("Indicateurs")
    available_indicators = get_strategy_ui_indicators(strategy_key)
    # Tous les indicateurs sont toujours affichés
    active_indicators: List[str] = available_indicators if available_indicators else []

    if available_indicators:
        st.sidebar.caption(f"📊 {len(available_indicators)} indicateur(s) : {', '.join(available_indicators)}")
    else:
        st.sidebar.caption("Aucun indicateur disponible.")

    # (Versioned Presets moved to bottom)

    st.sidebar.subheader("🔄 Mode d'exécution")

    if "optimization_mode" not in st.session_state:
        st.session_state.optimization_mode = "Grille de Paramètres"
    if "run_backtest_requested" not in st.session_state:
        st.session_state.run_backtest_requested = False
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    if "default_preset_applied" not in st.session_state:
        cpu_count = os.cpu_count() or 1
        optimal_workers = min(cpu_count, 64)  # ✅ max 64 pour gros CPU
        st.session_state["grid_n_workers"] = optimal_workers
        st.session_state["grid_worker_threads"] = 1
        st.session_state["gpu_n_workers"] = 1
        st.session_state["gpu_worker_threads"] = 1
        os.environ["BACKTEST_WORKER_THREADS"] = "1"
        st.session_state["default_preset_applied"] = True

    st.sidebar.markdown(MODE_BUTTON_CSS, unsafe_allow_html=True)

    for mode_name, icon, description in MODE_OPTIONS:
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
                width="stretch",
                type="primary" if is_active else "secondary",
                disabled=st.session_state.is_running,
            ):
                st.session_state.optimization_mode = mode_name
                st.rerun()

    optimization_mode = st.session_state.optimization_mode

    st.sidebar.caption(f"ℹ️ Mode actif: **{optimization_mode}**")

    action_slot = st.sidebar.container()

    # default_max_combos non utilisé - supprimé pour éviter warning
    unlimited_max_combos = 1_000_000_000_000
    default_workers_cpu = _env_int("BACKTEST_MAX_WORKERS", None)
    if default_workers_cpu is None:
        default_workers_cpu = _env_int("BACKTEST_WORKERS_CPU_OPTIMIZED", None)
    if default_workers_cpu is None:
        default_workers_cpu = _env_int("BACKTEST_WORKERS_GPU_OPTIMIZED", 40)
    default_llm_unload = _env_bool("UNLOAD_LLM_DURING_BACKTEST", True)
    default_worker_threads = _env_int("BACKTEST_WORKER_THREADS", 1)

    max_combos = unlimited_max_combos
    n_workers = default_workers_cpu

    # Configuration Optuna (intégrée dans Grille de Paramètres)
    use_optuna = False
    optuna_n_trials = 100
    optuna_sampler = "tpe"
    optuna_pruning = True
    optuna_metric = "sharpe_ratio"
    optuna_early_stop = 0  # 0 = désactivé par défaut

    if optimization_mode == "Grille de Paramètres":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Méthode d'exploration")

        use_optuna = st.sidebar.checkbox(
            "Utiliser Optuna (Bayésien) ⚡",
            value=False,
            help="Optuna explore intelligemment l'espace des paramètres (10-100x plus rapide que la grille exhaustive)",
        )

        if use_optuna:
            st.sidebar.caption("🎯 **Mode Bayésien** - Exploration intelligente")

            optuna_n_trials = st.sidebar.number_input(
                "Nombre de trials",
                min_value=10,
                max_value=10000,
                value=200,
                step=10,
                help="Nombre d'essais bayésiens (100-500 recommandé)",
            )

            optuna_sampler = st.sidebar.selectbox(
                "Algorithme",
                ["tpe", "cmaes", "random"],
                index=0,
                help="TPE: Rapide et efficace | CMA-ES: Pour espaces continus | Random: Baseline",
            )

            optuna_metric = st.sidebar.selectbox(
                "Métrique à optimiser",
                ["sharpe_ratio", "sortino_ratio", "total_return_pct", "profit_factor", "calmar_ratio"],
                index=0,
                help="Métrique principale pour l'optimisation",
            )

            optuna_pruning = st.sidebar.checkbox(
                "Pruning (arrêt précoce) ✂️",
                value=True,
                help="Abandonne les trials peu prometteurs pour accélérer",
            )

            # Early stop: 0 = désactivé, sinon patience en nombre de trials
            optuna_early_stop = st.sidebar.slider(
                "Early stop patience (0=désactivé)",
                min_value=0,
                max_value=max(200, optuna_n_trials),
                value=0,  # Désactivé par défaut pour ne pas interrompre prématurément
                help="Arrêt après N trials sans amélioration. 0 = désactivé (recommandé pour explorer complètement)",
            )

            n_workers = st.sidebar.slider(
                "Workers parallèles (CPU)",
                min_value=1,
                max_value=64,  # ✅ Augmenté pour gros CPU
                value=8,
                help="Nombre de trials évalués en parallèle",
            )

            st.sidebar.caption(f"⚡ {optuna_n_trials} trials × {n_workers} workers")
        else:
            st.sidebar.caption("🔢 **Mode Grille** - Exploration exhaustive")

            max_combos = unlimited_max_combos
            st.sidebar.caption("Limite de combinaisons: illimitée")

            grid_workers_default = max(1, min(default_workers_cpu, 64))  # ✅ max 64
            if "grid_n_workers" not in st.session_state:
                st.session_state["grid_n_workers"] = grid_workers_default
            else:
                try:
                    st.session_state["grid_n_workers"] = max(
                        1,
                        min(int(st.session_state["grid_n_workers"]), 64),  # ✅ max 64
                    )
                except (TypeError, ValueError):
                    st.session_state["grid_n_workers"] = grid_workers_default

            grid_threads_default = max(1, min(default_worker_threads, 16))  # ✅ max 16 (pas besoin de plus)
            if "grid_worker_threads" not in st.session_state:
                st.session_state["grid_worker_threads"] = grid_threads_default
            else:
                try:
                    st.session_state["grid_worker_threads"] = max(
                        1,
                        min(int(st.session_state["grid_worker_threads"]), 16),  # ✅ max 16
                    )
                except (TypeError, ValueError):
                    st.session_state["grid_worker_threads"] = grid_threads_default

            col_preset_1, col_preset_2 = st.sidebar.columns(2)
            with col_preset_1:
                if st.button("Preset 32 cœurs", key="preset_32_cores"):
                    st.session_state["grid_n_workers"] = 32
                    st.session_state["grid_worker_threads"] = 1
                    st.rerun()
            with col_preset_2:
                if st.button("Preset 64 cœurs", key="preset_64_cores"):
                    st.session_state["grid_n_workers"] = 64
                    st.session_state["grid_worker_threads"] = 1
                    st.rerun()

            # Initialisation session_state si nécessaire
            if "grid_n_workers" not in st.session_state:
                st.session_state["grid_n_workers"] = 32

            n_workers = st.sidebar.slider(
                "Workers parallèles (CPU)",
                min_value=1,
                max_value=64,  # ✅ Augmenté pour supporter les gros CPU (Threadripper, EPYC)
                help="24-32 recommandé pour 9950X. Données pré-chargées = initialisation rapide",
                key="grid_n_workers",
            )

            # Initialisation session_state si nécessaire
            if "grid_worker_threads" not in st.session_state:
                st.session_state["grid_worker_threads"] = 1

            worker_threads = st.sidebar.slider(
                "Threads par worker (CPU/BLAS)",
                min_value=1,
                max_value=16,
                step=1,
                key="grid_worker_threads",
                help=(
                    "Nombre de threads CPU par process. "
                    "Total ≈ workers × threads. "
                    "Recommandé: 1 si beaucoup de workers."
                ),
            )

            st.sidebar.caption(
                f"Total théorique: ~{n_workers * worker_threads} threads (workers × threads)"
            )
    llm_config = None
    llm_max_iterations = 10
    llm_use_walk_forward = True
    role_model_config = None
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
    llm_unload_during_backtest = default_llm_unload
    llm_model = None

    if optimization_mode == "🤖 Optimisation LLM":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Configuration LLM")

        st.sidebar.markdown("---")
        st.sidebar.caption("**⚙️ Paramètres d'exécution**")

        max_combos = unlimited_max_combos
        st.sidebar.caption("Limite de combinaisons LLM: illimitée")

        llm_workers_default = max(1, min(default_workers_cpu, 64))  # ✅ max 64
        n_workers = st.sidebar.slider(
            "Workers parallèles (CPU)",
            min_value=1,
            max_value=64,  # ✅ Augmenté pour gros CPU
            value=llm_workers_default,
            help="Nombre de backtests exécutés en parallèle (40 recommandé)",
            key="llm_n_workers",
        )

        st.sidebar.caption(
            f"🔧 Parallélisation: jusqu'à {n_workers} backtests simultanés"
        )
        st.sidebar.markdown("---")

        if not LLM_AVAILABLE:
            st.sidebar.error("❌ Module LLM non disponible")
            st.sidebar.caption(f"Erreur: {LLM_IMPORT_ERROR}")
        else:
            llm_provider = st.sidebar.selectbox(
                "Provider LLM",
                ["Ollama (Local)", "OpenAI"],
                help="Ollama = gratuit et local | OpenAI = API payante",
            )

            llm_use_multi_agent = st.sidebar.checkbox(
                "Mode multi-agents 👥",
                value=False,
                key="llm_use_multi_agent",
                help="Utiliser Analyst/Strategist/Critic/Validator",
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

                llm_use_multi_model = False
                if llm_use_multi_agent:
                    llm_use_multi_model = st.sidebar.checkbox(
                        "Multi-modeles par role",
                        value=False,
                        key="llm_use_multi_model",
                        help="Assigner differents modeles a chaque role d'agent",
                    )

                if llm_use_multi_model:
                    available_models_list = list_available_models()
                    available_model_names = [m.name for m in available_models_list]

                    llm_limit_small_models = st.sidebar.checkbox(
                        "Limiter selection aleatoire a <20B",
                        value=True,
                        key="llm_limit_small_models",
                        help="Filtre la liste par taille et exclut deepseek-r1:70b",
                    )
                    llm_limit_large_models = st.sidebar.checkbox(
                        "Limiter selection aleatoire a >=20B",
                        value=False,
                        key="llm_limit_large_models",
                        help="Filtre la liste par taille (>=20B uniquement)",
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
                            st.sidebar.warning("Aucun modele >=20B detecte.")
                    if effective_large_filter and not available_model_names:
                        st.sidebar.error(
                            "Selection >=20B activee mais aucun modele compatible."
                        )

                    st.sidebar.markdown("---")
                    st.sidebar.caption("**Configuration des modèles**")

                    # ===== GESTION DES PRESETS =====
                    from ui.model_presets import (
                        apply_preset_to_config,
                        delete_model_preset,
                        get_current_config_as_dict,
                        list_model_presets,
                        load_model_preset,
                        save_model_preset,
                    )

                    # Lister tous les presets
                    all_presets = list_model_presets()
                    preset_names = [p["name"] for p in all_presets]

                    # Selectbox pour choisir un preset
                    col1, col2 = st.sidebar.columns([3, 1])
                    with col1:
                        selected_preset = st.selectbox(
                            "Charger un preset",
                            options=["Aucun (manuel)"] + preset_names,
                            key="selected_model_preset",
                            help="Charge une configuration prédéfinie de modèles LLM"
                        )

                    with col2:
                        # Bouton pour appliquer le preset
                        if selected_preset != "Aucun (manuel)":
                            if st.button("⚡", key="apply_preset", help="Appliquer ce preset"):
                                preset = load_model_preset(selected_preset)
                                apply_preset_to_config(preset, get_global_model_config())
                                st.rerun()

                    # Expander pour sauvegarder/gérer les presets
                    with st.sidebar.expander("💾 Gérer les presets"):
                        user_presets = [p for p in all_presets if not p.get("builtin", False)]

                        # Tab pour organiser les actions
                        action_choice = st.radio(
                            "Action",
                            ["➕ Créer nouveau", "✏️ Modifier existant", "🗑️ Supprimer"],
                            key="preset_action",
                            horizontal=True
                        )

                        if action_choice == "➕ Créer nouveau":
                            st.markdown("**Créer un nouveau preset**")
                            new_preset_name = st.text_input(
                                "Nom du preset",
                                key="new_preset_name",
                                placeholder="Ex: Précis, Rapide, Test..."
                            )
                            st.caption("💡 Ajustez les modèles ci-dessous avant de sauvegarder")

                            if st.button("💾 Créer", key="create_preset"):
                                if new_preset_name.strip():
                                    try:
                                        current_config = get_current_config_as_dict(get_global_model_config())
                                        save_model_preset(new_preset_name.strip(), current_config["models"])
                                        st.success(f"✅ Preset '{new_preset_name}' créé")
                                        st.rerun()
                                    except ValueError as e:
                                        st.error(f"❌ {e}")
                                else:
                                    st.error("Nom de preset requis")

                        elif action_choice == "✏️ Modifier existant":
                            st.markdown("**Modifier un preset existant**")
                            if user_presets:
                                preset_to_modify = st.selectbox(
                                    "Preset à modifier",
                                    options=[p["name"] for p in user_presets],
                                    key="preset_to_modify"
                                )
                                st.caption("💡 Chargez le preset ci-dessus, ajustez les modèles, puis sauvegardez")

                                if st.button("💾 Sauvegarder modifications", key="update_preset"):
                                    try:
                                        current_config = get_current_config_as_dict(get_global_model_config())
                                        save_model_preset(preset_to_modify, current_config["models"])
                                        st.success(f"✅ Preset '{preset_to_modify}' mis à jour")
                                        st.rerun()
                                    except ValueError as e:
                                        st.error(f"❌ {e}")
                            else:
                                st.info("Aucun preset utilisateur à modifier")

                        elif action_choice == "🗑️ Supprimer":
                            st.markdown("**Supprimer un preset**")
                            if user_presets:
                                preset_to_delete = st.selectbox(
                                    "Preset à supprimer",
                                    options=[p["name"] for p in user_presets],
                                    key="preset_to_delete"
                                )
                                st.warning(f"⚠️ Supprimer '{preset_to_delete}' définitivement ?")

                                if st.button("🗑️ Confirmer suppression", key="delete_preset"):
                                    try:
                                        if delete_model_preset(preset_to_delete):
                                            st.success(f"✅ Preset '{preset_to_delete}' supprimé")
                                            st.rerun()
                                    except ValueError as e:
                                        st.error(f"❌ {e}")
                            else:
                                st.info("Aucun preset utilisateur à supprimer")

                    st.sidebar.markdown("---")
                    st.sidebar.caption("**Modeles par role d'agent**")
                    st.sidebar.caption("Rapide | Moyen | Lent")

                    # Checkbox pour pré-configuration optimale
                    use_optimal_config = st.sidebar.checkbox(
                        "Pré-config optimale",
                        value=False,
                        key="use_optimal_model_config",
                        help=(
                            "Active la configuration recommandée basée sur les benchmarks:\n"
                            "• Analyst → qwen2.5:14b (rapide)\n"
                            "• Strategist → gemma3:27b (équilibré)\n"
                            "• Critic → llama3.3-70b-optimized (puissant)\n"
                            "• Validator → llama3.3-70b-optimized (critique)"
                        ),
                    )

                    if use_optimal_config:
                        st.sidebar.info(
                            "💡 Configuration optimale activée. "
                            "Vous pouvez ajuster manuellement les sélections ci-dessous."
                        )

                    role_model_config = get_global_model_config()

                    def model_with_badge(name: str) -> str:
                        info = KNOWN_MODELS.get(name)
                        if info:
                            if info.category == ModelCategory.LIGHT:
                                return f"[L] {name}"
                            if info.category == ModelCategory.MEDIUM:
                                return f"[M] {name}"
                            return f"[H] {name}"
                        return name

                    model_options_display = [
                        model_with_badge(m) for m in available_model_names
                    ]
                    name_to_display = {
                        n: model_with_badge(n) for n in available_model_names
                    }
                    display_to_name = {v: k for k, v in name_to_display.items()}

                    use_single_model_for_roles = st.sidebar.checkbox(
                        "Même modèle pour tous les rôles",
                        value=False,
                        key="llm_single_model_for_roles",
                        help="Applique un seul modèle à Analyst/Strategist/Critic/Validator.",
                    )

                    single_model_selection = None
                    if use_single_model_for_roles:
                        if model_options_display:
                            default_model = (
                                role_model_config.analyst.models[0]
                                if role_model_config.analyst.models
                                else (available_model_names[0] if available_model_names else None)
                            )
                            default_display = name_to_display.get(
                                default_model, model_options_display[0]
                            )
                            default_index = (
                                model_options_display.index(default_display)
                                if default_display in model_options_display
                                else 0
                            )
                            single_model_selection = st.sidebar.selectbox(
                                "Modèle unique (tous rôles)",
                                model_options_display,
                                index=default_index,
                                key="llm_single_model_for_roles_name",
                                help="Ce modèle sera utilisé pour tous les rôles.",
                            )
                        else:
                            st.sidebar.warning(
                                "Aucun modèle disponible pour unifier les rôles."
                            )

                    st.sidebar.markdown("**Analyst** (analyse rapide)")

                    if selected_preset and selected_preset != "Aucun (manuel)":
                        # Charger le preset et utiliser ses modèles
                        preset = load_model_preset(selected_preset)
                        preset_models = preset["models"].get("analyst", [])
                        analyst_default_options = [
                            name_to_display.get(m, m)
                            for m in preset_models
                            if m in available_model_names
                        ]
                    elif use_optimal_config:
                        # Utiliser la config optimale
                        from ui.components.model_selector import get_optimal_config_for_role
                        optimal_analyst = get_optimal_config_for_role("analyst", available_model_names)
                        analyst_default_options = [
                            name_to_display.get(m, m) for m in optimal_analyst
                        ]
                    else:
                        # Comportement existant
                        analyst_defaults = [
                            name_to_display.get(m, m)
                            for m in role_model_config.analyst.models
                            if m in available_model_names
                        ]
                        analyst_default_options = (
                            analyst_defaults[:3] if analyst_defaults else model_options_display[:2]
                        )

                    if not model_options_display:
                        analyst_default_options = []

                    analyst_selection = st.sidebar.multiselect(
                        "Modeles Analyst",
                        model_options_display,
                        default=analyst_default_options,
                        key="analyst_models",
                        help="Modeles rapides recommandes pour l'analyse",
                    )

                    st.sidebar.markdown("**Strategist** (propositions)")

                    if selected_preset and selected_preset != "Aucun (manuel)":
                        # Charger le preset et utiliser ses modèles
                        preset = load_model_preset(selected_preset)
                        preset_models = preset["models"].get("strategist", [])
                        strategist_default_options = [
                            name_to_display.get(m, m)
                            for m in preset_models
                            if m in available_model_names
                        ]
                    elif use_optimal_config:
                        # Utiliser la config optimale
                        optimal_strategist = get_optimal_config_for_role("strategist", available_model_names)
                        strategist_default_options = [
                            name_to_display.get(m, m) for m in optimal_strategist
                        ]
                    else:
                        # Comportement existant
                        strategist_defaults = [
                            name_to_display.get(m, m)
                            for m in role_model_config.strategist.models
                            if m in available_model_names
                        ]
                        strategist_default_options = (
                            strategist_defaults[:3]
                            if strategist_defaults
                            else model_options_display[:2]
                        )

                    if not model_options_display:
                        strategist_default_options = []

                    strategist_selection = st.sidebar.multiselect(
                        "Modeles Strategist",
                        model_options_display,
                        default=strategist_default_options,
                        key="strategist_models",
                        help="Modeles moyens pour la creativite",
                    )

                    st.sidebar.markdown("**Critic** (evaluation critique)")

                    if selected_preset and selected_preset != "Aucun (manuel)":
                        # Charger le preset et utiliser ses modèles
                        preset = load_model_preset(selected_preset)
                        preset_models = preset["models"].get("critic", [])
                        critic_default_options = [
                            name_to_display.get(m, m)
                            for m in preset_models
                            if m in available_model_names
                        ]
                    elif use_optimal_config:
                        # Utiliser la config optimale
                        optimal_critic = get_optimal_config_for_role("critic", available_model_names)
                        critic_default_options = [
                            name_to_display.get(m, m) for m in optimal_critic
                        ]
                    else:
                        # Comportement existant
                        critic_defaults = [
                            name_to_display.get(m, m)
                            for m in role_model_config.critic.models
                            if m in available_model_names
                        ]
                        critic_default_options = (
                            critic_defaults[:3] if critic_defaults else model_options_display[:2]
                        )

                    if not model_options_display:
                        critic_default_options = []

                    critic_selection = st.sidebar.multiselect(
                        "Modeles Critic",
                        model_options_display,
                        default=critic_default_options,
                        key="critic_models",
                        help="Modeles puissants pour la reflexion",
                    )

                    st.sidebar.markdown("**Validator** (décision finale)")

                    if selected_preset and selected_preset != "Aucun (manuel)":
                        # Charger le preset et utiliser ses modèles
                        preset = load_model_preset(selected_preset)
                        preset_models = preset["models"].get("validator", [])
                        validator_default_options = [
                            name_to_display.get(m, m)
                            for m in preset_models
                            if m in available_model_names
                        ]
                    elif use_optimal_config:
                        # Utiliser la config optimale
                        optimal_validator = get_optimal_config_for_role("validator", available_model_names)
                        validator_default_options = [
                            name_to_display.get(m, m) for m in optimal_validator
                        ]
                    else:
                        # Comportement existant
                        validator_defaults = [
                            name_to_display.get(m, m)
                            for m in role_model_config.validator.models
                            if m in available_model_names
                        ]
                        validator_default_options = (
                            validator_defaults[:3]
                            if validator_defaults
                            else model_options_display[:2]
                        )

                    if not model_options_display:
                        validator_default_options = []

                    validator_selection = st.sidebar.multiselect(
                        "Modeles Validator",
                        model_options_display,
                        default=validator_default_options,
                        key="validator_models",
                        help="Modeles puissants pour decisions finales",
                    )

                    if use_single_model_for_roles and single_model_selection:
                        analyst_selection = [single_model_selection]
                        strategist_selection = [single_model_selection]
                        critic_selection = [single_model_selection]
                        validator_selection = [single_model_selection]

                    st.sidebar.markdown("---")
                    st.sidebar.caption("Modeles lourds")
                    heavy_after_iter = st.sidebar.number_input(
                        "Autoriser apres iteration N",
                        min_value=1,
                        max_value=20,
                        value=3,
                        help="Les modeles lourds ne seront utilises qu'apres cette iteration",
                    )

                    def _normalize_selection(selection: List[str]) -> List[str]:
                        names = [display_to_name.get(m, m) for m in selection]
                        return [n for n in names if n in available_model_names]

                    role_model_config.analyst.models = _normalize_selection(analyst_selection)
                    role_model_config.strategist.models = _normalize_selection(strategist_selection)
                    role_model_config.critic.models = _normalize_selection(critic_selection)
                    role_model_config.validator.models = _normalize_selection(validator_selection)

                    for assignment in [
                        role_model_config.analyst,
                        role_model_config.strategist,
                        role_model_config.critic,
                        role_model_config.validator,
                    ]:
                        assignment.allow_heavy_after_iteration = heavy_after_iter

                    set_global_model_config(role_model_config)

                    st.sidebar.info(
                        "Si plusieurs modeles sont selectionnes, "
                        "un sera choisi aleatoirement a chaque appel."
                    )

                    if role_model_config.analyst.models:
                        llm_model = role_model_config.analyst.models[0]
                    elif available_model_names:
                        llm_model = available_model_names[0]
                    elif effective_large_filter:
                        llm_model = None
                    else:
                        llm_model = "deepseek-r1:8b"

                else:
                    available_models = get_available_models_for_ui(
                        preferred_order=RECOMMENDED_FOR_STRATEGY
                    )

                    llm_model = st.sidebar.selectbox(
                        "Modèle Ollama",
                        available_models,
                        help="Modèles installés localement via Ollama",
                    )

                    if llm_model:
                        model_info = get_model_info(llm_model)
                        size = model_info["size_gb"]
                        desc = model_info["description"]
                        st.sidebar.caption(f"📦 ~{size} GB | {desc}")

                ollama_host = st.sidebar.text_input(
                    "URL Ollama",
                    value="http://localhost:11434",
                    help="Adresse du serveur Ollama",
                )
                if llm_model:
                    llm_config = LLMConfig(
                        provider=LLMProvider.OLLAMA,
                        model=llm_model,
                        ollama_host=ollama_host,
                    )
                else:
                    llm_config = None
            else:
                openai_key = st.sidebar.text_input(
                    "Clé API OpenAI",
                    type="password",
                    help="Votre clé API OpenAI",
                )
                llm_model = st.sidebar.selectbox(
                    "Modèle OpenAI",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    help="gpt-4o-mini recommandé pour coût/performance",
                )
                if openai_key:
                    llm_config = LLMConfig(
                        provider=LLMProvider.OPENAI,
                        model=llm_model,
                        api_key=openai_key,
                    )
                else:
                    st.sidebar.warning("⚠️ Clé API requise")

            st.sidebar.markdown("---")
            st.sidebar.caption("**Options d'optimisation**")

            llm_unlimited_iterations = st.sidebar.checkbox(
                "Itérations illimitées",
                value=True,
                key="llm_unlimited_iterations",
                help="Lance l'optimisation sans limite d'itérations (arrêt manuel requis)",
            )

            if llm_unlimited_iterations:
                llm_max_iterations = 0
                st.sidebar.caption("∞ itérations (arrêt manuel)")
            else:
                llm_max_iterations = st.sidebar.slider(
                    "Max itérations",
                    min_value=3,
                    max_value=50,
                    value=10,
                    help="Nombre max de cycles d'amélioration",
                )

            walk_forward_enabled = True
            walk_forward_reason = ""

            df_cached = st.session_state.get("ohlcv_df")
            if df_cached is not None and not df_cached.empty:
                data_duration_days = (df_cached.index[-1] - df_cached.index[0]).days
                data_duration_months = data_duration_days / 30.44

                if data_duration_months < 6:
                    walk_forward_enabled = False
                    walk_forward_reason = (
                        "⚠️ Walk-Forward désactivé "
                        f"(durée: {data_duration_months:.1f} mois < 6 mois requis)"
                    )
                else:
                    walk_forward_reason = (
                        f"✅ Walk-Forward disponible (durée: {data_duration_months:.1f} mois)"
                    )

            if walk_forward_reason:
                if walk_forward_enabled:
                    st.sidebar.caption(walk_forward_reason)
                else:
                    st.sidebar.warning(walk_forward_reason)

            llm_use_walk_forward = st.sidebar.checkbox(
                "Walk-Forward Validation",
                value=walk_forward_enabled,
                disabled=not walk_forward_enabled,
                help=(
                    "Anti-overfitting: valide sur données hors-échantillon "
                    "(nécessite >6 mois de données)"
                ),
            )

            llm_unload_during_backtest = st.sidebar.checkbox(
                "Décharger LLM du GPU",
                value=default_llm_unload,
                help=(
                    "Libère la VRAM pendant les backtests pour améliorer les performances. "
                    "Peut être désactivé en mode CPU-only."
                ),
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
                max_iter_label = "∞" if llm_max_iterations <= 0 else str(llm_max_iterations)
                st.sidebar.caption(
                    "Agents: Analyst/Strategist/Critic/Validator | "
                    f"Max iterations: {max_iter_label}"
                )
            else:
                max_iter_label = "∞" if llm_max_iterations <= 0 else str(llm_max_iterations)
                st.sidebar.caption(
                    f"Agent autonome | Max iterations: {max_iter_label}"
                )

    # ======================== GPU ACCELERATION ========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Accélération GPU")
    st.sidebar.info(
        "Mode CPU-only: GPU désactivé.\n\n"
        "• Numba JIT + cache RAM utilisés\n"
        "• VRAM libérée pour autres usages"
    )
    os.environ["BACKTEST_USE_GPU"] = "0"
    os.environ["BACKTEST_GPU_QUEUE_ENABLED"] = "0"

    st.sidebar.subheader("🔧 Paramètres")

    param_mode = "range" if optimization_mode == "Grille de Paramètres" else "single"

    params: Dict[str, Any] = {}
    param_ranges: Dict[str, Any] = {}
    param_specs: Dict[str, Any] = {}
    strategy_class = get_strategy(strategy_key)
    strategy_instance = None

    if strategy_class:
        temp_strategy = strategy_class()
        strategy_instance = temp_strategy
        param_specs = temp_strategy.parameter_specs or {}
        label_overrides: Dict[str, str] = {}

        if strategy_key == "bollinger_best_longe_3i":
            label_overrides = {
                "entry_level": "Entrée",
                "tp_level": "Sortie_gagnante",
                "sl_level": "Stop-loss",
                "bb_std": "Bollinger_amplitude",
                "bb_period": "Bollinger_signal",
            }

        if param_specs:
            validation_errors = []

            for param_name, spec in param_specs.items():
                if not getattr(spec, "optimize", True):
                    continue

                if param_mode == "single":
                    value = create_param_range_selector(
                        param_name,
                        strategy_key,
                        mode="single",
                        spec=spec,
                        label=label_overrides.get(param_name),
                    )
                    if value is not None:
                        params[param_name] = value

                        is_valid, error = validate_param(param_name, value)
                        if not is_valid:
                            validation_errors.append(error)
                else:
                    range_data = create_param_range_selector(
                        param_name,
                        strategy_key,
                        mode="range",
                        spec=spec,
                        label=label_overrides.get(param_name),
                    )
                    if range_data is not None:
                        param_ranges[param_name] = range_data
                        if spec is not None:
                            params[param_name] = spec.default
                        else:
                            params[param_name] = PARAM_CONSTRAINTS[param_name]["default"]
                        # DEBUG: Afficher les ranges générés
                        print(f"[DEBUG] param_ranges[{param_name}] = {range_data}")

            if validation_errors:
                for err in validation_errors:
                    st.sidebar.error(err)

            # DEBUG: Afficher le résumé des param_ranges
            print(f"[DEBUG] param_ranges final = {list(param_ranges.keys())}")
            print(f"[DEBUG] Total paramètres optimisables: {sum(1 for s in param_specs.values() if getattr(s, 'optimize', True))}")

            if param_mode == "range" and param_ranges:
                st.sidebar.markdown("---")
                stats = compute_search_space_stats(
                    param_ranges,
                    max_combinations=max_combos,
                )

                if stats.is_continuous:
                    st.sidebar.info("ℹ️ Espace continu détecté")
                elif stats.has_overflow:
                    st.sidebar.warning(
                        f"⚠️ {stats.total_combinations:,} combinaisons (limite: {max_combos:,})"
                    )
                    st.sidebar.caption("Réduisez les plages ou augmentez le step")
                else:
                    st.sidebar.success(
                        f"✅ {stats.total_combinations:,} combinaisons à tester"
                    )

                with st.sidebar.expander("📊 Détail par paramètre"):
                    for pname, pcount in stats.per_param_counts.items():
                        st.caption(f"• {pname}: {pcount} valeurs")
            else:
                st.sidebar.caption("📊 Mode simple: 1 combinaison")
    else:
        st.sidebar.error(f"Stratégie '{strategy_key}' non trouvée")

    st.sidebar.subheader("💰 Trading")

    # Checkbox pour activer/désactiver le leverage
    leverage_enabled = st.sidebar.checkbox(
        "� Activer le leverage",
        value=False,  # Désactivé par défaut = leverage forcé à 1
        key="leverage_enabled",
        help="Si décoché, leverage=1 (sans effet de levier). Recommandé pour tests sûrs.",
    )

    if leverage_enabled:
        leverage = create_param_range_selector("leverage", "trading", mode="single")
        params["leverage"] = leverage
    else:
        leverage = 1.0
        params["leverage"] = 1.0
        st.sidebar.caption("_Leverage désactivé → forcé à 1×_")

    initial_capital = st.sidebar.number_input(
        "Capital Initial ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Capital de départ (1,000 - 1,000,000)",
    )

    # Liste des paramètres désactivés (pour transmission au backtest)
    disabled_params: List[str] = []
    if not leverage_enabled:
        disabled_params.append("leverage")

    # Ajouter les indicateurs non cochés à disabled_params (info seulement)
    if available_indicators:
        unchecked_indicators = [ind for ind in available_indicators if ind not in active_indicators]
        if unchecked_indicators:
            st.sidebar.caption(f"_Indicateurs masqués: {', '.join(unchecked_indicators)}_")

    st.sidebar.subheader("💾 Versioned Presets")

    versioned_presets = list_strategy_versions(strategy_key)

    if "_sync_preset_version" in st.session_state:
        st.session_state["versioned_preset_version"] = st.session_state.pop(
            "_sync_preset_version"
        )
    if "_sync_preset_name" in st.session_state:
        st.session_state["versioned_preset_name"] = st.session_state.pop(
            "_sync_preset_name"
        )

    last_saved = st.session_state.pop("versioned_preset_last_saved", None)
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
            p for p in versioned_presets if (p.metadata or {}).get("version") == selected_version
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

        if st.sidebar.button("⬇️ Charger versioned preset", key="load_versioned_preset"):
            try:
                loaded_preset = load_strategy_version(
                    strategy_name=strategy_key,
                    version=selected_version,
                    preset_name=selected_preset_name,
                )
                apply_versioned_preset(loaded_preset, strategy_key)
                st.session_state["loaded_versioned_preset"] = loaded_preset.to_dict()
                st.sidebar.success("Versioned preset loaded")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Failed to load preset: {exc}")
    else:
        st.sidebar.caption("No versioned presets found.")

    render_saved_runs_panel(
        st.session_state.get("last_run_result"),
        strategy_key,
        symbol,
        timeframe,
    )

    # Multi-sweep lists (symbols et timeframes déjà définis par multiselect)
    # strategy_keys et all_params/ranges/specs pour toutes les stratégies sélectionnées
    strategy_keys = [strategy_options[name] for name in strategy_names]
    all_params = {}
    all_param_ranges = {}
    all_param_specs = {}

    # Pour la première stratégie, utiliser les paramètres configurés via l'UI
    all_params[strategy_key] = params
    all_param_ranges[strategy_key] = param_ranges
    all_param_specs[strategy_key] = param_specs

    # Pour les autres stratégies, utiliser les paramètres par défaut
    for name in strategy_names[1:]:
        other_strategy_key = strategy_options[name]
        other_strategy_class = get_strategy(other_strategy_key)
        if other_strategy_class:
            other_instance = other_strategy_class()
            other_specs = other_instance.parameter_specs or {}
            other_params = {}
            other_ranges = {}

            for param_name, spec in other_specs.items():
                if not getattr(spec, "optimize", True):
                    continue
                # Utiliser les valeurs par défaut
                other_params[param_name] = spec.default
                if param_mode == "range":
                    # Créer un range basique pour l'optimisation
                    other_ranges[param_name] = {
                        "min": spec.min_val,
                        "max": spec.max_val,
                        "step": spec.step,
                    }

            all_params[other_strategy_key] = other_params
            all_param_ranges[other_strategy_key] = other_ranges
            all_param_specs[other_strategy_key] = other_specs

    draft_state = SidebarState(
        debug_enabled=debug_enabled,
        symbol=symbol,
        timeframe=timeframe,
        use_date_filter=use_date_filter,
        start_date=start_date,
        end_date=end_date,
        available_tokens=available_tokens,
        available_timeframes=available_timeframes,
        strategy_key=strategy_key,
        strategy_name=strategy_name,
        strategy_info=strategy_info,
        strategy_instance=strategy_instance,
        params=params,
        param_ranges=param_ranges,
        param_specs=param_specs,
        active_indicators=active_indicators,
        optimization_mode=optimization_mode,
        max_combos=max_combos,
        n_workers=n_workers,
        # Multi-sweep lists
        symbols=symbols,
        timeframes=timeframes,
        strategy_keys=strategy_keys,
        all_params=all_params,
        all_param_ranges=all_param_ranges,
        all_param_specs=all_param_specs,
        # Optuna
        use_optuna=use_optuna,
        optuna_n_trials=optuna_n_trials,
        optuna_sampler=optuna_sampler,
        optuna_pruning=optuna_pruning,
        optuna_metric=optuna_metric,
        optuna_early_stop=optuna_early_stop,
        llm_config=llm_config,
        llm_model=llm_model,
        llm_use_multi_agent=llm_use_multi_agent,
        role_model_config=role_model_config,
        llm_max_iterations=llm_max_iterations,
        llm_use_walk_forward=llm_use_walk_forward,
        llm_unload_during_backtest=llm_unload_during_backtest,
        llm_compare_enabled=llm_compare_enabled,
        llm_compare_auto_run=llm_compare_auto_run,
        llm_compare_strategies=llm_compare_strategies,
        llm_compare_tokens=llm_compare_tokens,
        llm_compare_timeframes=llm_compare_timeframes,
        llm_compare_metric=llm_compare_metric,
        llm_compare_aggregate=llm_compare_aggregate,
        llm_compare_max_runs=llm_compare_max_runs,
        llm_compare_use_preset=llm_compare_use_preset,
        llm_compare_generate_report=llm_compare_generate_report,
        initial_capital=initial_capital,
        leverage=leverage,
        leverage_enabled=leverage_enabled,
        disabled_params=disabled_params,
    )

    applied_state = _apply_config_guard(draft_state)
    pending = st.session_state.get("config_pending_changes", False)

    run_label_map = {
        "Backtest Simple": "🚀 Lancer le Backtest",
        "Grille de Paramètres": "🧪 Lancer le Sweep",
        "🤖 Optimisation LLM": "🧠 Lancer l'itération LLM",
    }
    run_label = run_label_map.get(
        st.session_state.optimization_mode,
        "🚀 Lancer le Backtest",
    )

    def _apply_pending_config() -> None:
        st.session_state["applied_config_signature"] = st.session_state.get(
            "draft_config_signature"
        )
        st.session_state["applied_sidebar_state"] = draft_state
        st.session_state["config_pending_changes"] = False

    with action_slot:
        st.markdown("---")
        st.caption("Actions")

        col_load, col_run = st.columns(2)
        with col_load:
            if st.button(
                "⬇️ Charger données",
                key="load_ohlcv_action",
                disabled=st.session_state.is_running,
                width="stretch",
            ):
                if pending:
                    _apply_pending_config()
                df_loaded, msg = load_selected_data(
                    symbol, timeframe, start_date, end_date
                )
                if df_loaded is None:
                    st.error(f"Erreur chargement: {msg}")
                else:
                    st.success(f"Données chargées: {msg}")

        with col_run:
            if st.button(
                run_label,
                key="run_sidebar_action",
                type="primary",
                disabled=st.session_state.is_running,
                width="stretch",
            ):
                if pending:
                    _apply_pending_config()
                st.session_state.run_backtest_requested = True
                st.rerun()

        if pending:
            st.warning(
                "⚠️ Modifications non appliquées (seront appliquées au lancement/chargement)"
            )
        else:
            st.caption("✅ Configuration prête.")

    return applied_state
