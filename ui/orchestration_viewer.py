"""
UI Component : Visualiseur de Logs d'Orchestration LLM
=======================================================

Composant Streamlit pour afficher les logs d'orchestration des agents LLM
de manière interactive et structurée.
"""

from typing import List, Optional
from html import escape
import streamlit as st
import pandas as pd
from datetime import datetime

from agents.orchestration_logger import (
    OrchestrationLogger,
    OrchestrationLogEntry,
    OrchestrationActionType,
    OrchestrationStatus,
)


def render_orchestration_logs(
    orchestration_logger: OrchestrationLogger,
    show_filters: bool = True,
    max_entries: int = 50
):
    """
    Affiche les logs d'orchestration avec filtres et visualisation.

    Args:
        orchestration_logger: Instance du logger d'orchestration
        show_filters: Si True, affiche les filtres interactifs
        max_entries: Nombre maximum d'entrées à afficher
    """
    st.markdown("### 🤖 Journal d'Orchestration LLM")

    if len(orchestration_logger.logs) == 0:
        st.info("Aucun log d'orchestration disponible")
        return

    # Métriques rapides
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Session ID", orchestration_logger.session_id[-8:])

    with col2:
        st.metric("Itérations", orchestration_logger.current_iteration)

    with col3:
        st.metric("Total Logs", len(orchestration_logger.logs))

    with col4:
        # Compter les actions complétées
        completed = sum(
            1 for log in orchestration_logger.logs
            if log.status == OrchestrationStatus.COMPLETED
        )
        st.metric("Complétés", completed)

    # Filtres
    if show_filters:
        st.markdown("---")
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            # Filtre par agent
            agents = list(set(log.agent for log in orchestration_logger.logs if log.agent))
            agents_filter = st.multiselect(
                "Filtrer par Agent",
                ["Tous"] + agents,
                default=["Tous"]
            )

        with col_f2:
            # Filtre par type d'action
            action_types = list(set(log.action_type for log in orchestration_logger.logs))
            action_filter = st.multiselect(
                "Filtrer par Action",
                ["Tous"] + [at.value for at in action_types],
                default=["Tous"]
            )

        with col_f3:
            # Filtre par itération
            iterations = list(set(log.iteration for log in orchestration_logger.logs))
            iteration_filter = st.multiselect(
                "Filtrer par Itération",
                ["Toutes"] + [f"Itération {i}" for i in sorted(iterations)],
                default=["Toutes"]
            )

        # Appliquer les filtres
        filtered_logs = orchestration_logger.logs

        if "Tous" not in agents_filter:
            filtered_logs = [log for log in filtered_logs if log.agent in agents_filter]

        if "Tous" not in action_filter:
            action_values = [at.value for at in action_types if at.value in action_filter]
            filtered_logs = [log for log in filtered_logs if log.action_type.value in action_values]

        if "Toutes" not in iteration_filter:
            selected_iterations = [
                int(it.split(" ")[1]) for it in iteration_filter if it != "Toutes"
            ]
            filtered_logs = [log for log in filtered_logs if log.iteration in selected_iterations]

    else:
        filtered_logs = orchestration_logger.logs

    st.markdown("---")

    # Affichage des logs
    _render_logs_timeline(filtered_logs, max_entries)


def _render_logs_timeline(logs: List[OrchestrationLogEntry], max_entries: int):
    """Affiche les logs sous forme de timeline."""
    st.markdown("#### 📋 Timeline des Actions")

    # Limiter le nombre de logs
    display_logs = logs[-max_entries:] if len(logs) > max_entries else logs

    if len(logs) > max_entries:
        st.caption(f"Affichage des {max_entries} derniers logs sur {len(logs)} total")

    # Grouper par itération
    logs_by_iteration = {}
    for log in display_logs:
        iteration = log.iteration
        if iteration not in logs_by_iteration:
            logs_by_iteration[iteration] = []
        logs_by_iteration[iteration].append(log)

    # Afficher par itération
    for iteration in sorted(logs_by_iteration.keys(), reverse=True):
        iteration_logs = logs_by_iteration[iteration]

        with st.expander(f"🔄 **Itération {iteration}** ({len(iteration_logs)} actions)", expanded=(iteration == max(logs_by_iteration.keys()))):
            for log in iteration_logs:
                _render_log_entry(log)


def _render_log_entry(log: OrchestrationLogEntry):
    """Affiche une entrée de log individuelle."""
    # Emoji et couleur selon le statut
    emoji = log._get_emoji()
    status_color = _get_status_color(log.status)
    text_color = _get_contrast_text_color(status_color)

    # Timestamp
    try:
        timestamp = datetime.fromisoformat(log.timestamp).strftime("%H:%M:%S")
    except:
        timestamp = log.timestamp[:8]  # Fallback

    # Agent badge
    model_name = ""
    if isinstance(log.details, dict):
        model_name = log.details.get("model") or ""
    agent_label = log.agent or ""
    if model_name:
        agent_label = f"{agent_label} · {model_name}" if agent_label else model_name
    agent_badge = (
        f"<strong style='color: {text_color};'>[{escape(agent_label)}]</strong>"
        if agent_label
        else ""
    )

    # Action type
    action_type = escape(log.action_type.value.replace("_", " ").title())

    # Construire la ligne principale
    timestamp_html = (
        "<code style='color: {text_color}; background-color: rgba(0,0,0,0.08); "
        "padding: 1px 4px; border-radius: 3px;'>"
        "{timestamp}</code>"
    ).format(text_color=text_color, timestamp=escape(timestamp))
    parts = [emoji, timestamp_html, agent_badge, action_type]
    main_line = " ".join(part for part in parts if part)

    # Afficher
    st.markdown(
        "<div style='background-color: {status_color}; color: {text_color}; "
        "padding: 8px; border-radius: 5px; margin: 5px 0;'>"
        "{main_line}</div>".format(
            status_color=status_color,
            text_color=text_color,
            main_line=main_line,
        ),
        unsafe_allow_html=True,
    )

    # Détails si disponibles
    if log.details:
        with st.container():
            _render_log_details(log)


def _render_log_details(log: OrchestrationLogEntry):
    """Affiche les détails d'un log."""
    details = log.details

    # Stratégie
    if "strategy" in details:
        st.caption(f"   Stratégie: `{details['strategy']}`")

    if "old_strategy" in details and "new_strategy" in details:
        st.caption(f"   `{details['old_strategy']}` → `{details['new_strategy']}`")

    # Indicateur
    if "indicator" in details:
        st.caption(f"   Indicateur: `{details['indicator']}`")

    if "old_values" in details and "new_values" in details:
        st.caption(f"   Anciens: {details['old_values']}")
        st.caption(f"   Nouveaux: {details['new_values']}")

    # Paramètres
    if "params" in details:
        params = details['params']
        if isinstance(params, dict):
            param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
            st.caption(f"   Paramètres: {param_str}...")

    # Résultats
    if "results" in details:
        results = details['results']
        if isinstance(results, dict):
            if "sharpe" in results:
                st.caption(f"   📊 Sharpe: {results['sharpe']:.3f}")
            if "pnl" in results:
                st.caption(f"   💰 PnL: {results['pnl']:.2f}")

    # Raison
    if "reason" in details:
        st.caption(f"   Raison: _{details['reason']}_")

    # Message / Erreur
    if "message" in details:
        st.caption(f"   📝 {details['message']}")

    if "error" in details:
        st.error(f"   ❌ Erreur: {details['error']}")


def _get_status_color(status: OrchestrationStatus) -> str:
    """Retourne la couleur de fond selon le statut."""
    colors = {
        OrchestrationStatus.COMPLETED: "#d4edda",  # Vert clair
        OrchestrationStatus.FAILED: "#f8d7da",     # Rouge clair
        OrchestrationStatus.VALIDATED: "#d1ecf1",   # Bleu clair
        OrchestrationStatus.REJECTED: "#f8d7da",    # Rouge clair
        OrchestrationStatus.IN_PROGRESS: "#fff3cd", # Jaune clair
        OrchestrationStatus.PENDING: "#e7e7e7",     # Gris clair
    }
    return colors.get(status, "#ffffff")


def _get_contrast_text_color(background_hex: str) -> str:
    """Retourne une couleur de texte lisible selon la luminance du fond."""
    hex_color = background_hex.lstrip("#")
    if len(hex_color) != 6:
        return "#1f2933"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return "#1f2933"
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "#111827" if luminance > 140 else "#f9fafb"


def render_orchestration_summary_table(orchestration_logger: OrchestrationLogger):
    """Affiche un tableau récapitulatif des actions par agent."""
    st.markdown("#### 📊 Récapitulatif par Agent")

    # Compter les actions par agent
    agent_actions = {}
    for log in orchestration_logger.logs:
        if log.agent:
            if log.agent not in agent_actions:
                agent_actions[log.agent] = {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "pending": 0
                }

            agent_actions[log.agent]["total"] += 1

            if log.status == OrchestrationStatus.COMPLETED:
                agent_actions[log.agent]["completed"] += 1
            elif log.status == OrchestrationStatus.FAILED:
                agent_actions[log.agent]["failed"] += 1
            elif log.status == OrchestrationStatus.PENDING:
                agent_actions[log.agent]["pending"] += 1

    # Créer le DataFrame
    if agent_actions:
        df = pd.DataFrame.from_dict(agent_actions, orient="index")
        df = df.reset_index()
        df.columns = ["Agent", "Total", "Complétés", "Échoués", "En Attente"]

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune action d'agent enregistrée")


def render_orchestration_metrics(orchestration_logger: OrchestrationLogger):
    """Affiche les métriques clés de l'orchestration."""
    st.markdown("#### 📈 Métriques d'Orchestration")

    # Compter les backtests
    backtests_launched = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.BACKTEST_LAUNCH)
    )
    backtests_completed = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.BACKTEST_COMPLETE)
    )
    backtests_failed = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.BACKTEST_FAILED)
    )

    # Compter les changements de stratégie
    strategy_changes = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.STRATEGY_MODIFICATION)
    )

    # Compter les changements d'indicateurs
    indicator_changes = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.INDICATOR_VALUES_CHANGE)
    )
    indicator_adds = len(
        orchestration_logger.get_logs_by_type(OrchestrationActionType.INDICATOR_ADD)
    )

    # Afficher
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Backtests Lancés", backtests_launched)

    with col2:
        st.metric("Backtests Réussis", backtests_completed)

    with col3:
        st.metric("Backtests Échoués", backtests_failed)

    with col4:
        st.metric("Changements Stratégie", strategy_changes)

    with col5:
        st.metric("Modifications Indicateurs", indicator_changes + indicator_adds)


def render_full_orchestration_viewer(
    orchestration_logger: OrchestrationLogger,
    max_entries: int = 50,
    show_filters: bool = True,
):
    """Affiche le visualiseur complet d'orchestration."""
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📋 Timeline", "📊 Résumé", "📈 Métriques"])

    with tab1:
        render_orchestration_logs(
            orchestration_logger,
            show_filters=show_filters,
            max_entries=max_entries,
        )

    with tab2:
        render_orchestration_summary_table(orchestration_logger)

    with tab3:
        render_orchestration_metrics(orchestration_logger)


__all__ = [
    "render_orchestration_logs",
    "render_orchestration_summary_table",
    "render_orchestration_metrics",
    "render_full_orchestration_viewer",
]
