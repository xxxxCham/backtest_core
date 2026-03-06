"""Module-ID: ui.exec_tabs — Tabs mode d'exécution (pilot v0)

Ce module rend les onglets de sélection du mode d'exécution dans la page
principale (hors sidebar). Il est appelé depuis app.py après
render_setup_previews() et avant render_main().
"""
from __future__ import annotations

import streamlit as st

from ui.constants import MODE_OPTIONS
from ui.state import SidebarState


def _init_exec_tabs_state() -> None:
    """Initialisation idempotente — sûre à appeler à chaque rerun."""
    defaults: dict = {
        "optimization_mode": "Grille de Paramètres",
        "grid_worker_threads": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_exec_tabs(state: SidebarState) -> None:
    """Affiche les onglets de mode d'exécution dans la page principale.

    Appelé depuis app.py APRÈS render_setup_previews(), AVANT render_main().
    Un clic sur un onglet inactif bascule le mode et relance un rerun.

    Args:
        state: État sidebar courant (lecture seule ici).
    """
    _init_exec_tabs_state()

    tab_labels = [f"{icon} {name}" for name, icon, _desc in MODE_OPTIONS]
    tabs = st.tabs(tab_labels)

    for i, ((mode_name, icon, desc), tab) in enumerate(zip(MODE_OPTIONS, tabs)):
        with tab:
            if state.optimization_mode != mode_name:
                if st.button(
                    f"→ Activer {icon} {mode_name}",
                    key=f"tab_activate_{i}",
                    help=desc,
                    disabled=st.session_state.get("is_running", False),
                ):
                    st.session_state.optimization_mode = mode_name
                    st.rerun()
            else:
                st.caption(f"✅ Mode actif — {desc}")
            # Commit B : if mode_name == "Backtest Simple": render_backtest_tab(state)
            # Commit C : if mode_name == "Grille de Paramètres": render_grid_tab(state)
