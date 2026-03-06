"""Module-ID: ui.exec_tabs — Tabs mode d'exécution (pilot v0)

Ce module rend les onglets de sélection du mode d'exécution dans la page
principale (hors sidebar). Il est appelé depuis app.py après
render_setup_previews() et avant render_main().
"""
from __future__ import annotations

import streamlit as st

from ui.constants import MODE_OPTIONS
from ui.state import SidebarState

# ── Clés session_state exposées aux onglets (lues ensuite par sidebar.py) ──
EXEC_GRID_USE_OPTUNA = "exec_grid_use_optuna"
EXEC_GRID_N_TRIALS = "exec_grid_n_trials"
EXEC_GRID_SAMPLER = "exec_grid_sampler"
EXEC_GRID_METRIC = "exec_grid_metric"
EXEC_GRID_PRUNING = "exec_grid_pruning"
EXEC_GRID_EARLY_STOP = "exec_grid_early_stop"


def _init_exec_tabs_state() -> None:
    """Initialisation idempotente — sûre à appeler à chaque rerun."""
    defaults: dict = {
        "optimization_mode": "Grille de Paramètres",
        "grid_worker_threads": 1,
        EXEC_GRID_USE_OPTUNA: False,
        EXEC_GRID_N_TRIALS: 200,
        EXEC_GRID_SAMPLER: "tpe",
        EXEC_GRID_METRIC: "sharpe_ratio",
        EXEC_GRID_PRUNING: True,
        EXEC_GRID_EARLY_STOP: 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Commit B — Onglet Backtest Simple
# ─────────────────────────────────────────────────────────────────────────────

def _render_backtest_tab(state: SidebarState) -> None:
    """Contenu de l'onglet Backtest Simple."""
    st.markdown("#### 📊 Backtest Simple")
    st.caption("Teste **1 combinaison** de paramètres — résultat immédiat, idéal pour valider une hypothèse.")

    if state.strategy_key:
        col1, col2, col3 = st.columns(3)
        col1.metric("Stratégie", state.strategy_key)
        col2.metric("Symbole", state.symbol or "—")
        col3.metric("Timeframe", state.timeframe or "—")

        with st.expander("Paramètres appliqués", expanded=False):
            st.json(state.params)

        if state.use_walk_forward:
            st.info(f"🔬 Walk-Forward actif — {state.wfa_n_folds} folds, train {state.wfa_train_ratio:.0%}")
    else:
        st.info("Sélectionnez une stratégie dans la sidebar pour commencer.")

    st.markdown("---")
    if st.button(
        "🚀 Lancer le Backtest",
        type="primary",
        key="run_btn_backtest_tab",
        disabled=st.session_state.get("is_running", False),
    ):
        st.session_state.run_backtest_requested = True
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Commit C — Onglet Grille / Optuna
# ─────────────────────────────────────────────────────────────────────────────

def _render_grid_tab(state: SidebarState) -> None:
    """Contenu de l'onglet Grille de Paramètres / Optuna.

    Widgets écrits dans session_state (via key=).  sidebar.py les lit ensuite
    pour alimenter SidebarState avant chaque run.
    """
    st.markdown("#### 🔢 Grille de Paramètres / Optuna")
    n_workers = int(st.session_state.get("ui_n_workers", 32))

    use_optuna = st.checkbox(
        "⚡ Utiliser Optuna (Bayésien)",
        key=EXEC_GRID_USE_OPTUNA,
        help="Explore intelligemment l'espace — 10‑100× plus rapide que la grille exhaustive.",
    )

    if use_optuna:
        st.caption("🎯 **Mode Bayésien** — exploration intelligente")
        col_a, col_b = st.columns(2)
        with col_a:
            st.number_input(
                "Nombre de trials",
                min_value=10,
                max_value=10_000,
                step=10,
                key=EXEC_GRID_N_TRIALS,
                help="Recommandé : 100-500",
            )
            st.selectbox(
                "Algorithme",
                ["tpe", "cmaes", "random"],
                key=EXEC_GRID_SAMPLER,
                help="TPE : rapide | CMA-ES : espaces continus | Random : baseline",
            )
        with col_b:
            st.selectbox(
                "Métrique à optimiser",
                ["sharpe_ratio", "sortino_ratio", "total_return_pct", "profit_factor", "calmar_ratio"],
                key=EXEC_GRID_METRIC,
            )
            st.checkbox(
                "Pruning ✂️ (arrêt précoce)",
                key=EXEC_GRID_PRUNING,
                help="Abandonne les trials peu prometteurs — accélère la recherche.",
            )
        n_trials_val = int(st.session_state.get(EXEC_GRID_N_TRIALS, 200))
        st.slider(
            "Early stop patience (0 = désactivé)",
            min_value=0,
            max_value=max(200, n_trials_val),
            key=EXEC_GRID_EARLY_STOP,
            help="Arrêt après N trials sans amélioration.",
        )
        st.caption(f"⚡ {n_trials_val} trials × {n_workers} workers")
    else:
        st.caption("🔢 **Mode Grille exhaustive** — explore tous les points min/max/step")
        st.markdown("---")
        # CRITICAL: key='grid_worker_threads' conservée verbatim — lue par render_main
        _default_threads = max(1, min(int(st.session_state.get("grid_worker_threads", 1)), 16))
        if "grid_worker_threads" not in st.session_state:
            st.session_state["grid_worker_threads"] = _default_threads
        worker_threads = st.slider(
            "Threads par worker (CPU/BLAS)",
            min_value=1,
            max_value=16,
            step=1,
            key="grid_worker_threads",
            help="Total ≈ workers × threads. Recommandé : 1 si beaucoup de workers.",
        )
        st.caption(f"Total théorique : ~{n_workers * worker_threads} threads")

    # ── Commit D — Run button câblé dans l'onglet ──
    st.markdown("---")
    if st.button(
        "🧪 Lancer le Sweep",
        type="primary",
        key="run_btn_grid_tab",
        disabled=st.session_state.get("is_running", False),
    ):
        st.session_state.run_backtest_requested = True
        st.rerun()


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
            # Contenu spécifique au mode actif
                if mode_name == "Backtest Simple":
                    _render_backtest_tab(state)
                elif mode_name == "Grille de Paramètres":
                    _render_grid_tab(state)
                # LLM / Builder : config toujours dans sidebar pour l'instant
