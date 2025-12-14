"""
Backtest Core - LLM Thinking Stream Viewer
===========================================

Composant pour afficher les pensées des agents LLM en temps réel.

Features:
- Stream de raisonnement en direct
- Catégorisation par type (thinking, conclusion, decision, error)
- Affichage timestamp et agent
- Historique limité pour performance

Usage:
    >>> from ui.components.thinking_viewer import ThinkingStreamViewer
    >>>
    >>> # Dans Streamlit
    >>> viewer = ThinkingStreamViewer()
    >>> viewer.add_thought("Analyst", "qwq:32b", "Analyzing metrics...", "thinking")
    >>> viewer.render()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal

import streamlit as st

ThoughtCategory = Literal["thinking", "conclusion", "decision", "error"]


@dataclass
class ThoughtEntry:
    """Entrée de pensée dans le stream."""

    timestamp: datetime
    agent_name: str
    model: str
    thought: str
    category: ThoughtCategory


class ThinkingStreamViewer:
    """
    Visualisation stream de pensées des LLMs en temps réel.

    Affiche:
    - Raisonnement de chaque agent
    - Conclusions et métriques
    - Décisions prises
    - Erreurs rencontrées

    Example:
        >>> viewer = ThinkingStreamViewer(container_key="llm_stream")
        >>> viewer.add_thought(
        ...     agent_name="Analyst",
        ...     model="deepseek-r1:32b",
        ...     thought="Sharpe actuel: 1.23 - Volatilité acceptable",
        ...     category="thinking"
        ... )
        >>> viewer.render()
    """

    def __init__(self, container_key: str = "thinking_stream"):
        """
        Initialise le viewer.

        Args:
            container_key: Clé unique pour session state Streamlit
        """
        self.container_key = container_key
        self._init_session_state()

    def _init_session_state(self):
        """Initialise variables session state."""
        key_stream = f"{self.container_key}_stream"
        if key_stream not in st.session_state:
            st.session_state[key_stream] = []

    def add_thought(
        self,
        agent_name: str,
        model: str,
        thought: str,
        category: ThoughtCategory = "thinking",
    ) -> None:
        """
        Ajoute une pensée au stream.

        Args:
            agent_name: Nom de l'agent (Analyst, Strategist, Critic, Validator)
            model: Modèle LLM utilisé
            thought: Contenu de la pensée
            category: Type de pensée ('thinking', 'conclusion', 'decision', 'error')

        Example:
            >>> viewer.add_thought(
            ...     "Strategist",
            ...     "qwq:32b",
            ...     "Testing bb_period=25 with k_sl=2.0",
            ...     "thinking"
            ... )
        """
        key_stream = f"{self.container_key}_stream"

        entry = ThoughtEntry(
            timestamp=datetime.now(),
            agent_name=agent_name,
            model=model,
            thought=thought,
            category=category,
        )

        st.session_state[key_stream].append(entry)

        # Limiter historique à 100 entrées pour éviter surcharge mémoire
        if len(st.session_state[key_stream]) > 100:
            st.session_state[key_stream] = st.session_state[key_stream][-100:]

    def clear(self) -> None:
        """Vide le stream de pensées."""
        key_stream = f"{self.container_key}_stream"
        st.session_state[key_stream] = []

    def get_thought_count(self) -> int:
        """Retourne le nombre de pensées dans le stream."""
        key_stream = f"{self.container_key}_stream"
        return len(st.session_state.get(key_stream, []))

    def render(self, max_entries: int = 20, show_header: bool = True) -> None:
        """
        Affiche le stream de pensées.

        Args:
            max_entries: Nombre maximum de pensées à afficher
            show_header: Afficher le header "Stream de Pensées LLM"

        Example:
            >>> viewer.render(max_entries=10, show_header=True)
        """
        key_stream = f"{self.container_key}_stream"
        stream: List[ThoughtEntry] = st.session_state.get(key_stream, [])

        if show_header:
            st.markdown("### 💭 Stream de Pensées LLM")

        if not stream:
            st.info("💭 Aucune activité mentale pour le moment...")
            st.caption(
                "Les pensées des agents LLM s'afficheront ici pendant l'optimisation"
            )
            return

        # Afficher les dernières entrées (ordre inverse = plus récent en haut)
        for entry in reversed(stream[-max_entries:]):
            self._render_thought_entry(entry)

    def _render_thought_entry(self, entry: ThoughtEntry) -> None:
        """
        Affiche une entrée de pensée avec style approprié.

        Args:
            entry: Entrée de pensée à afficher
        """
        # Style selon catégorie
        category_config = {
            "thinking": {"emoji": "🤔", "alert_type": "info"},
            "conclusion": {"emoji": "💡", "alert_type": "success"},
            "decision": {"emoji": "🎯", "alert_type": "warning"},
            "error": {"emoji": "❌", "alert_type": "error"},
        }

        config = category_config.get(entry.category, {"emoji": "💬", "alert_type": "info"})
        emoji = config["emoji"]
        alert_type = config["alert_type"]

        # Format timestamp
        timestamp_str = entry.timestamp.strftime("%H:%M:%S")

        # En-tête avec timestamp, agent et modèle
        with st.container():
            st.markdown(
                f"**{timestamp_str}** | {emoji} **{entry.agent_name}** (`{entry.model}`)"
            )

            # Contenu avec style approprié
            if alert_type == "info":
                st.info(entry.thought)
            elif alert_type == "success":
                st.success(entry.thought)
            elif alert_type == "warning":
                st.warning(entry.thought)
            elif alert_type == "error":
                st.error(entry.thought)

            st.divider()


def render_thinking_stream(
    container_key: str = "thinking_stream",
    max_entries: int = 20,
    show_header: bool = True,
) -> ThinkingStreamViewer:
    """
    Fonction helper pour créer et afficher le viewer en une ligne.

    Args:
        container_key: Clé unique pour session state
        max_entries: Nombre maximum de pensées à afficher
        show_header: Afficher le header

    Returns:
        ThinkingStreamViewer: Instance du viewer pour ajouter des pensées

    Example:
        >>> # Dans Streamlit:
        >>> viewer = render_thinking_stream()
        >>> # Plus tard, pour ajouter une pensée:
        >>> viewer.add_thought("Analyst", "qwq:32b", "Analyzing...", "thinking")
    """
    viewer = ThinkingStreamViewer(container_key=container_key)
    viewer.render(max_entries=max_entries, show_header=show_header)
    return viewer


__all__ = [
    "ThinkingStreamViewer",
    "ThoughtEntry",
    "ThoughtCategory",
    "render_thinking_stream",
]
