"""
Module-ID: ui.deep_trace_viewer

Purpose: Visualiseur Deep Trace détaillé pour orchestration LLM - timeline, inspecteur, propositions, state machine, métriques.

Role in pipeline: visualization / debugging

Key components: render_deep_trace(), timeline, metrics, state inspector

Inputs: OrchestrationLogger instance

Outputs: Interface Streamlit multi-onglets (timeline, inspector, metrics, etc.)

Dependencies: streamlit, agents.orchestration_logger

Conventions: Onglets: Timeline, Inspector, Proposals, State Machine, Metrics

Read-if: Deep debugging orchestration LLM ou inspection états.

Skip-if: Pas d'agents LLM ou monitoring minimal suffisant.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from agents.orchestration_logger import (
    OrchestrationActionType,
    OrchestrationLogEntry,
    OrchestrationLogger,
    OrchestrationStatus,
)

# ============================================================================
# UTILITAIRES
# ============================================================================


def _format_float(value: Any, precision: int) -> Optional[str]:
    """Formate un float de facon sure (retourne None si invalide)."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return f"{parsed:.{precision}f}"


def _format_percent(value: Any, precision: int = 2) -> Optional[str]:
    """Formate un pourcentage de facon sure (retourne None si invalide)."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return f"{parsed:.{precision}%}"


def _get_event_emoji(action_type: OrchestrationActionType) -> str:
    """Retourne l'emoji approprié pour un type d'événement."""
    emoji_map = {
        OrchestrationActionType.RUN_START: "🚀",
        OrchestrationActionType.RUN_END: "🏁",
        OrchestrationActionType.PHASE_START: "📍",
        OrchestrationActionType.STATE_ENTER: "➡️",
        OrchestrationActionType.STATE_CHANGE: "🔄",
        OrchestrationActionType.AGENT_EXECUTE_START: "🤖",
        OrchestrationActionType.AGENT_EXECUTE_END: "✅",
        OrchestrationActionType.PROPOSALS_GENERATED: "💡",
        OrchestrationActionType.PROPOSAL_TEST_STARTED: "🧪",
        OrchestrationActionType.PROPOSAL_TEST_ENDED: "📊",
        OrchestrationActionType.VALIDATOR_DECISION: "⚖️",
        OrchestrationActionType.BACKTEST_START: "▶️",
        OrchestrationActionType.BACKTEST_END: "⏹️",
        OrchestrationActionType.WARNING: "⚠️",
        OrchestrationActionType.ERROR: "❌",
        OrchestrationActionType.CONFIG_VALID: "✔️",
        OrchestrationActionType.CONFIG_INVALID: "❌",
    }
    return emoji_map.get(action_type, "•")


def _get_event_color(action_type: OrchestrationActionType, status: OrchestrationStatus) -> str:
    """Retourne la couleur de fond pour un événement."""
    # Priorité au statut
    if status == OrchestrationStatus.FAILED:
        return "#f8d7da"  # Rouge clair
    elif status == OrchestrationStatus.COMPLETED:
        return "#d4edda"  # Vert clair

    # Sinon, couleur par type
    color_map = {
        OrchestrationActionType.RUN_START: "#e0f7fa",
        OrchestrationActionType.RUN_END: "#c8e6c9",
        OrchestrationActionType.ERROR: "#f8d7da",
        OrchestrationActionType.WARNING: "#fff3cd",
        OrchestrationActionType.AGENT_EXECUTE_START: "#e3f2fd",
        OrchestrationActionType.AGENT_EXECUTE_END: "#c8e6c9",
        OrchestrationActionType.VALIDATOR_DECISION: "#fff9c4",
        OrchestrationActionType.BACKTEST_END: "#f3e5f5",
    }
    return color_map.get(action_type, "#f5f5f5")


def _format_timestamp(ts_str: str) -> str:
    """Formate un timestamp ISO en HH:MM:SS."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Millisecondes
    except Exception:
        return ts_str[:12] if len(ts_str) > 12 else ts_str


def _get_role_badge_color(role: Optional[str]) -> str:
    """Retourne la couleur du badge de rôle."""
    if not role:
        return "#9e9e9e"
    role_colors = {
        "analyst": "#2196f3",
        "strategist": "#4caf50",
        "critic": "#ff9800",
        "validator": "#9c27b0",
    }
    return role_colors.get(role.lower(), "#9e9e9e")


# ============================================================================
# TIMELINE COMPLÈTE
# ============================================================================

def render_timeline_panel(logger: OrchestrationLogger, filters: Dict[str, Any]):
    """
    Affiche la timeline complète des événements avec filtres.

    Args:
        logger: OrchestrationLogger instance
        filters: Dict avec 'iteration', 'agent', 'event_type', 'level'
    """
    st.markdown("### 📋 Timeline des Événements")

    # Appliquer les filtres
    filtered_logs = logger.logs

    if filters.get("iteration") and "Toutes" not in filters["iteration"]:
        iterations = [int(x.split()[1]) for x in filters["iteration"] if x != "Toutes"]
        filtered_logs = [log for log in filtered_logs if log.iteration in iterations]

    if filters.get("agent") and "Tous" not in filters["agent"]:
        filtered_logs = [log for log in filtered_logs if log.agent in filters["agent"]]

    if filters.get("event_type") and "Tous" not in filters["event_type"]:
        event_types = [OrchestrationActionType(et) for et in filters["event_type"]]
        filtered_logs = [log for log in filtered_logs if log.action_type in event_types]

    if filters.get("level"):
        level = filters["level"]
        if level == "ERROR":
            filtered_logs = [
                log
                for log in filtered_logs
                if log.action_type == OrchestrationActionType.ERROR
                or log.status == OrchestrationStatus.FAILED
            ]
        elif level == "WARNING":
            filtered_logs = [
                log
                for log in filtered_logs
                if log.action_type == OrchestrationActionType.WARNING
            ]

    if not filtered_logs:
        st.info("Aucun événement ne correspond aux filtres sélectionnés")
        return

    st.caption(f"Affichage de {len(filtered_logs)} événements sur {len(logger.logs)} total")

    # Grouper par itération
    logs_by_iteration = {}
    for log in filtered_logs:
        iteration = log.iteration
        if iteration not in logs_by_iteration:
            logs_by_iteration[iteration] = []
        logs_by_iteration[iteration].append(log)

    # Afficher par itération (ordre inverse)
    for iteration in sorted(logs_by_iteration.keys(), reverse=True):
        iteration_logs = logs_by_iteration[iteration]

        with st.expander(
            f"🔄 **Itération {iteration}** ({len(iteration_logs)} événements)",
            expanded=(iteration == max(logs_by_iteration.keys()))
        ):
            for log in iteration_logs:
                _render_timeline_entry(log)


def _render_timeline_entry(log: OrchestrationLogEntry):
    """Affiche une entrée de timeline."""
    emoji = _get_event_emoji(log.action_type)
    color = _get_event_color(log.action_type, log.status)
    timestamp = _format_timestamp(log.timestamp)

    # Badge de rôle
    role_badge = ""
    if log.agent:
        role_color = _get_role_badge_color(log.agent)
        role_badge = f'<span style="background-color: {role_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 8px;"><strong>{log.agent.upper()}</strong></span>'

    # Type d'événement
    event_type = log.action_type.value.replace("_", " ").title()

    # Ligne principale
    main_html = f"""
    <div style="background-color: {color}; padding: 8px 12px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #666;">
        <span style="font-size: 1.2em;">{emoji}</span>
        <code style="background-color: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 3px; margin: 0 8px;">{timestamp}</code>
        <strong>{event_type}</strong>
        {role_badge}
    </div>
    """
    st.markdown(main_html, unsafe_allow_html=True)

    # Détails si disponibles
    if log.details:
        _render_event_details(log)


def _render_event_details(log: OrchestrationLogEntry):
    """Affiche les détails d'un événement."""
    details = log.details

    # Extraire les champs importants
    important_fields = []

    # Modèle LLM
    if "model" in details:
        important_fields.append(f"📦 Modèle: `{details['model']}`")

    # Latence
    if "latency_ms" in details:
        latency = details["latency_ms"]
        latency_str = _format_float(latency, 0) or str(latency)
        important_fields.append(f"⏱️ Latence: {latency_str}ms")

    # Métriques
    if "sharpe" in details or "sharpe_ratio" in details:
        sharpe = details.get("sharpe") or details.get("sharpe_ratio")
        sharpe_str = _format_float(sharpe, 3) or "N/A"
        important_fields.append(f"📊 Sharpe: {sharpe_str}")

    if "total_return" in details:
        ret = details["total_return"]
        ret_str = _format_percent(ret, 2)
        if ret_str is None:
            ret_str = str(ret) if ret is not None else "N/A"
        important_fields.append(f"💰 Return: {ret_str}")

    if "max_drawdown" in details:
        dd = details["max_drawdown"]
        dd_str = _format_percent(dd, 2)
        if dd_str is None:
            dd_str = str(dd) if dd is not None else "N/A"
        important_fields.append(f"📉 Drawdown: {dd_str}")

    # Décision
    if "decision" in details:
        important_fields.append(f"⚖️ Décision: **{details['decision']}**")

    # Propositions
    if "count" in details:
        important_fields.append(f"💡 Nombre: {details['count']}")

    # Message/Erreur
    if "message" in details:
        msg = details["message"]
        if log.action_type == OrchestrationActionType.ERROR:
            important_fields.append(f"❌ {msg}")
        elif log.action_type == OrchestrationActionType.WARNING:
            important_fields.append(f"⚠️ {msg}")
        else:
            important_fields.append(f"📝 {msg}")

    # Afficher les champs importants
    if important_fields:
        for field in important_fields:
            st.caption(f"   {field}")

    # Bouton pour afficher le JSON complet
    with st.expander("🔍 Détails complets (JSON)", expanded=False):
        st.json(details, expanded=False)


# ============================================================================
# INSPECTEUR LLM
# ============================================================================

def render_llm_inspector_panel(logger: OrchestrationLogger):
    """
    Affiche l'inspecteur des échanges LLM par rôle.
    """
    st.markdown("### 🤖 Inspecteur LLM")

    # Récupérer tous les événements d'agents
    agent_events = [
        log for log in logger.logs
        if log.action_type in [
            OrchestrationActionType.AGENT_EXECUTE_START,
            OrchestrationActionType.AGENT_EXECUTE_END,
        ]
    ]

    if not agent_events:
        st.info("Aucun événement LLM enregistré")
        return

    # Grouper par rôle
    events_by_role = {}
    for log in agent_events:
        role = log.agent or "unknown"
        if role not in events_by_role:
            events_by_role[role] = []
        events_by_role[role].append(log)

    # Onglets par rôle
    roles = sorted(events_by_role.keys())
    if len(roles) == 1:
        _render_llm_role_details(roles[0], events_by_role[roles[0]])
    else:
        tabs = st.tabs([f"🎭 {role.capitalize()}" for role in roles])
        for tab, role in zip(tabs, roles):
            with tab:
                _render_llm_role_details(role, events_by_role[role])


def _render_llm_role_details(role: str, events: List[OrchestrationLogEntry]):
    """Affiche les détails LLM pour un rôle."""
    # Statistiques globales
    total_calls = len([e for e in events if e.action_type == OrchestrationActionType.AGENT_EXECUTE_END])
    successful = len([e for e in events if e.action_type == OrchestrationActionType.AGENT_EXECUTE_END and e.details.get("success")])

    total_latency = sum([
        e.details.get("latency_ms", 0)
        for e in events
        if e.action_type == OrchestrationActionType.AGENT_EXECUTE_END
    ])
    avg_latency = total_latency / total_calls if total_calls > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Appels", total_calls)
    with col2:
        st.metric("Réussis", successful)
    with col3:
        st.metric("Latence Moyenne", f"{avg_latency:.0f}ms")

    st.markdown("---")

    # Liste des appels
    st.markdown("#### 📞 Historique des Appels")

    # Filtrer uniquement les événements END (qui contiennent le résumé)
    end_events = [e for e in events if e.action_type == OrchestrationActionType.AGENT_EXECUTE_END]

    for i, event in enumerate(reversed(end_events), 1):
        details = event.details
        timestamp = _format_timestamp(event.timestamp)
        model = details.get("model", "inconnu")
        latency = details.get("latency_ms", 0)
        success = details.get("success", False)
        iteration = event.iteration

        status_emoji = "✅" if success else "❌"

        with st.expander(
            f"{status_emoji} **Appel #{i}** - Itération {iteration} - {timestamp} ({latency}ms)",
            expanded=False
        ):
            st.markdown(f"**Modèle:** `{model}`")
            st.markdown(f"**Itération:** {iteration}")
            st.markdown(f"**Timestamp:** {timestamp}")
            st.markdown(f"**Latence:** {latency}ms")
            st.markdown(f"**Statut:** {'Réussi ✅' if success else 'Échoué ❌'}")

            # Détails complets
            st.markdown("**Métadonnées complètes:**")
            st.json(details, expanded=False)


# ============================================================================
# PROPOSITIONS & TESTS
# ============================================================================

def render_proposals_panel(logger: OrchestrationLogger):
    """
    Affiche le panneau Propositions & Tests.
    """
    st.markdown("### 💡 Propositions & Tests")

    # Récupérer les événements de propositions
    proposal_events = [
        log for log in logger.logs
        if log.action_type in [
            OrchestrationActionType.PROPOSALS_GENERATED,
            OrchestrationActionType.PROPOSAL_TEST_STARTED,
            OrchestrationActionType.PROPOSAL_TEST_ENDED,
        ]
    ]

    if not proposal_events:
        st.info("Aucune proposition enregistrée")
        return

    # Compter les propositions générées vs testées
    generated_events = [e for e in proposal_events if e.action_type == OrchestrationActionType.PROPOSALS_GENERATED]
    test_ended_events = [e for e in proposal_events if e.action_type == OrchestrationActionType.PROPOSAL_TEST_ENDED]

    total_generated = sum([e.details.get("count", 0) for e in generated_events])
    total_tested = len(test_ended_events)
    tested_successful = len([e for e in test_ended_events if e.details.get("tested", False)])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Propositions Générées", total_generated)
    with col2:
        st.metric("Propositions Testées", total_tested)
    with col3:
        st.metric("Tests Réussis", tested_successful)

    st.markdown("---")

    # Tableau des tests
    if test_ended_events:
        st.markdown("#### 📊 Résultats des Tests")

        test_data = []
        for event in test_ended_events:
            details = event.details
            # Récupérer les valeurs (0.0 est une valeur valide, ne pas afficher "N/A")
            sharpe = details.get("sharpe")
            total_return = details.get("total_return")

            test_data.append({
                "ID": details.get("proposal_id", "N/A"),
                "Itération": event.iteration,
                "Testé": "✅" if details.get("tested") else "❌",
                "Sharpe": f"{sharpe:.3f}" if sharpe is not None else "N/A",
                "Return": f"{total_return:.2%}" if total_return is not None else "N/A",
                "Timestamp": _format_timestamp(event.timestamp),
            })

        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun test de proposition enregistré")


# ============================================================================
# STATE MACHINE
# ============================================================================

def render_state_machine_panel(logger: OrchestrationLogger):
    """
    Affiche le panneau State Machine.
    """
    st.markdown("### 🔄 State Machine")

    # Récupérer les événements d'états
    state_events = [
        log for log in logger.logs
        if log.action_type in [
            OrchestrationActionType.STATE_ENTER,
            OrchestrationActionType.STATE_CHANGE,
        ]
    ]

    if not state_events:
        st.info("Aucun événement d'état enregistré")
        return

    # État actuel (dernier STATE_ENTER)
    current_state = "UNKNOWN"
    if state_events:
        last_enter = next((e for e in reversed(logger.logs) if e.action_type == OrchestrationActionType.STATE_ENTER), None)
        if last_enter:
            current_state = last_enter.details.get("state", "UNKNOWN")

    # Nombre de transitions
    state_changes = [e for e in state_events if e.action_type == OrchestrationActionType.STATE_CHANGE]
    total_transitions = len(state_changes)

    # Itération courante
    current_iteration = logger.current_iteration

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("État Actuel", current_state)
    with col2:
        st.metric("Itération Courante", current_iteration)
    with col3:
        st.metric("Total Transitions", total_transitions)

    st.markdown("---")

    # Historique des transitions
    st.markdown("#### 📜 Historique des Transitions")

    if state_changes:
        transitions_data = []
        for event in reversed(state_changes):
            details = event.details
            transitions_data.append({
                "Itération": event.iteration,
                "De": details.get("state_from", "?"),
                "Vers": details.get("state_to", "?"),
                "Timestamp": _format_timestamp(event.timestamp),
            })

        df = pd.DataFrame(transitions_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune transition enregistrée")


# ============================================================================
# MÉTRIQUES GLOBALES
# ============================================================================

def render_metrics_panel(logger: OrchestrationLogger):
    """
    Affiche le panneau Métriques Globales de la session.
    """
    st.markdown("### 📈 Métriques Globales Session")

    # Compter les différents types d'événements
    llm_calls = len([
        e for e in logger.logs
        if e.action_type == OrchestrationActionType.AGENT_EXECUTE_END
    ])

    backtests_done = len([
        e for e in logger.logs
        if e.action_type == OrchestrationActionType.BACKTEST_END
        and e.details.get("success")
    ])

    errors_count = len([
        e for e in logger.logs
        if e.action_type == OrchestrationActionType.ERROR
        or e.status == OrchestrationStatus.FAILED
    ])

    warnings_count = len([
        e for e in logger.logs
        if e.action_type == OrchestrationActionType.WARNING
    ])

    # Temps total (si disponible)
    next((e for e in logger.logs if e.action_type == OrchestrationActionType.RUN_START), None)
    run_end = next((e for e in reversed(logger.logs) if e.action_type == OrchestrationActionType.RUN_END), None)

    total_time_s = 0
    if run_end and "total_time_s" in run_end.details:
        total_time_s = run_end.details["total_time_s"]

    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total LLM Calls", llm_calls)
    with col2:
        st.metric("Total Backtests", backtests_done)
    with col3:
        st.metric("Temps Total", f"{total_time_s:.1f}s" if total_time_s > 0 else "N/A")
    with col4:
        st.metric("⚠️ Warnings", warnings_count)
    with col5:
        st.metric("❌ Errors", errors_count)

    st.markdown("---")

    # Meilleur résultat (si disponible)
    st.markdown("#### 🏆 Meilleur Résultat")

    # Chercher dans les événements iteration_recorded
    iteration_events = [
        e for e in logger.logs
        if e.action_type == OrchestrationActionType.ITERATION_RECORDED
    ]

    if iteration_events:
        best_sharpe = max([e.details.get("sharpe", -999) for e in iteration_events if e.details.get("sharpe") is not None], default=None)

        if best_sharpe is not None and best_sharpe > -999:
            best_event = next((e for e in iteration_events if e.details.get("sharpe") == best_sharpe), None)
            if best_event:
                details = best_event.details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Sharpe", f"{best_sharpe:.3f}")
                with col2:
                    ret = details.get("total_return", 0)
                    st.metric("Return", f"{ret:.2%}" if ret else "N/A")
                with col3:
                    dd = details.get("max_drawdown", 0)
                    st.metric("Max Drawdown", f"{dd:.2%}" if dd else "N/A")
            else:
                st.info("Données de meilleur résultat non disponibles")
        else:
            st.info("Aucune métrique Sharpe enregistrée")
    else:
        st.info("Aucune itération enregistrée")


# ============================================================================
# FILTRES
# ============================================================================

def render_filters_sidebar(logger: OrchestrationLogger) -> Dict[str, Any]:
    """
    Affiche les filtres dans la sidebar et retourne les valeurs sélectionnées.

    Returns:
        Dict avec les filtres sélectionnés
    """
    st.sidebar.markdown("### 🔍 Filtres")

    # Filtre par itération
    iterations = sorted(set(log.iteration for log in logger.logs))
    iteration_filter = st.sidebar.multiselect(
        "Itération",
        ["Toutes"] + [f"Itération {i}" for i in iterations],
        default=["Toutes"]
    )

    # Filtre par agent
    agents = sorted(set(log.agent for log in logger.logs if log.agent))
    agent_filter = st.sidebar.multiselect(
        "Agent",
        ["Tous"] + agents,
        default=["Tous"]
    )

    # Filtre par type d'événement
    event_types = sorted(set(log.action_type.value for log in logger.logs))
    event_type_filter = st.sidebar.multiselect(
        "Type d'événement",
        ["Tous"] + event_types,
        default=["Tous"]
    )

    # Filtre par niveau (INFO/WARNING/ERROR)
    level_filter = st.sidebar.selectbox(
        "Niveau",
        ["TOUS", "ERROR", "WARNING"],
        index=0
    )

    return {
        "iteration": iteration_filter,
        "agent": agent_filter,
        "event_type": event_type_filter,
        "level": level_filter if level_filter != "TOUS" else None,
    }


# ============================================================================
# SESSION SELECTOR & LOADER
# ============================================================================

def render_session_selector() -> Optional[OrchestrationLogger]:
    """
    Affiche un sélecteur de session et permet de charger des traces.

    Returns:
        OrchestrationLogger chargé ou None
    """
    st.sidebar.markdown("### 📂 Chargement de Session")

    # Découvrir les sessions disponibles dans runs/
    runs_dir = Path("runs")
    if not runs_dir.exists():
        st.sidebar.info("Aucune session disponible (répertoire runs/ inexistant)")
        return None

    # Lister les sessions (dossiers avec trace.jsonl)
    available_sessions = []
    for session_dir in runs_dir.iterdir():
        if session_dir.is_dir():
            trace_file = session_dir / "trace.jsonl"
            if trace_file.exists():
                available_sessions.append(session_dir.name)

    if not available_sessions:
        st.sidebar.info("Aucune session disponible")

        # Option d'upload manuel
        uploaded_file = st.sidebar.file_uploader(
            "Charger un fichier trace (JSON/JSONL)",
            type=["json", "jsonl"],
            key="trace_uploader"
        )

        if uploaded_file:
            try:
                # Sauvegarder temporairement
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.read())

                # Charger
                logger = OrchestrationLogger.load_from_file(temp_path)
                st.sidebar.success(f"✅ Session {logger.session_id} chargée")
                return logger
            except Exception as e:
                st.sidebar.error(f"❌ Erreur de chargement: {e}")
                return None

        return None

    # Sélecteur de session
    selected_session = st.sidebar.selectbox(
        "Sélectionner une session",
        ["-- Sélectionnez --"] + sorted(available_sessions, reverse=True),
        index=0
    )

    if selected_session == "-- Sélectionnez --":
        return None

    # Charger la session
    trace_file = runs_dir / selected_session / "trace.jsonl"
    try:
        logger = OrchestrationLogger.load_from_file(trace_file)
        st.sidebar.success(f"✅ Session chargée: {selected_session}")
        return logger
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de chargement: {e}")
        return None


# ============================================================================
# EXPORT
# ============================================================================

def render_export_panel(logger: OrchestrationLogger, filters: Dict[str, Any]):
    """
    Affiche un panneau d'export des logs filtrés.
    """
    st.markdown("### 💾 Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Exporter JSON complet"):
            export_path = Path(f"orchestration_export_{logger.session_id}.json")
            logger.save_to_file(export_path)
            st.success(f"✅ Exporté vers {export_path}")

    with col2:
        if st.button("📥 Exporter JSONL"):
            export_path = Path(f"orchestration_export_{logger.session_id}.jsonl")
            logger.save_to_jsonl(export_path)
            st.success(f"✅ Exporté vers {export_path}")


# ============================================================================
# VIEWER PRINCIPAL
# ============================================================================

def render_deep_trace_viewer(logger: Optional[OrchestrationLogger] = None):
    """
    Point d'entrée principal pour le Deep Trace Viewer.

    Args:
        logger: Instance OrchestrationLogger (peut être None, auquel cas on charge depuis sélecteur)
    """
    st.markdown("## 🔍 Orchestration Deep Trace")

    # Si pas de logger fourni, essayer de charger depuis sélecteur
    if logger is None:
        logger = render_session_selector()

    if logger is None:
        st.info("👈 Sélectionnez une session dans la sidebar pour commencer")
        return

    # Informations de session
    st.markdown(f"**Session:** `{logger.session_id}`")
    st.markdown(f"**Total événements:** {len(logger.logs)}")
    st.markdown(f"**Itérations:** {logger.current_iteration}")

    st.markdown("---")

    # Filtres dans la sidebar
    filters = render_filters_sidebar(logger)

    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Timeline",
        "🤖 Inspecteur LLM",
        "💡 Propositions",
        "🔄 State Machine",
        "📈 Métriques",
        "💾 Export"
    ])

    with tab1:
        render_timeline_panel(logger, filters)

    with tab2:
        render_llm_inspector_panel(logger)

    with tab3:
        render_proposals_panel(logger)

    with tab4:
        render_state_machine_panel(logger)

    with tab5:
        render_metrics_panel(logger)

    with tab6:
        render_export_panel(logger, filters)


__all__ = [
    "render_deep_trace_viewer",
]
