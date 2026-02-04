"""
Exemple d'intégration du pattern st.form() dans l'UI Streamlit.

USAGE:
    streamlit run ui/main_with_form.py

Ce fichier montre comment intégrer le nouveau pattern de configuration
avec formulaire pour éviter les reloads inutiles.

Architecture:
    1. Sidebar: Formulaire de configuration (config_form.py)
    2. Main: Bouton Run + exécution avec config figée
    3. Preview: Affichage léger sans chargement données

Avantages:
    - Pas de reload à chaque changement de slider
    - Config figée pour l'exécution
    - Preview instantanée (pas de chargement lourd)
    - UX améliorée
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd

# Import du nouveau module de configuration
from ui.config_form import (
    render_minimal_config_form,
    get_frozen_config,
    reset_validation,
)
from ui.helpers import (
    load_selected_data,
    safe_run_backtest,
    show_status,
)
from ui.context import BacktestEngine


def render_run_button() -> bool:
    """
    Affiche le bouton Run dans la zone principale.

    Returns:
        True si le bouton est cliqué
    """
    # Vérifier si config validée
    cfg_validated = st.session_state.get("cfg_validated", False)

    if not cfg_validated:
        st.warning("⚠️ Validez d'abord la configuration dans la sidebar")
        return False

    # Bouton Run
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_clicked = st.button(
            "🚀 Lancer le Backtest",
            type="primary",
            use_container_width=True,
            disabled=not cfg_validated
        )

    return run_clicked


def execute_backtest_with_frozen_config(cfg_frozen: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Exécute le backtest avec une configuration figée.

    Args:
        cfg_frozen: Configuration immutable

    Returns:
        Résultats du backtest ou None si erreur
    """
    try:
        # === PHASE 1: CHARGEMENT DONNÉES ===
        with st.spinner(f"📥 Chargement données {cfg_frozen['symbol']} {cfg_frozen['timeframe']}..."):
            start_load = time.time()

            df = load_selected_data(
                symbol=cfg_frozen["symbol"],
                timeframe=cfg_frozen["timeframe"],
                use_date_filter=cfg_frozen.get("use_date_filter", False),
                start_date=cfg_frozen.get("start_date"),
                end_date=cfg_frozen.get("end_date"),
            )

            if df is None or df.empty:
                show_status("error", "Échec chargement données")
                return None

            load_duration = time.time() - start_load
            st.success(f"✅ Données chargées: {len(df):,} bougies ({load_duration:.2f}s)")

        # === PHASE 2: EXÉCUTION BACKTEST ===
        with st.spinner("⚙️ Exécution backtest..."):
            start_bt = time.time()

            # Créer l'engine
            engine = BacktestEngine(initial_capital=cfg_frozen.get("initial_capital", 100000.0))

            # Paramètres pour exécution simple (pas de sweep pour l'instant)
            params = {}  # TODO: gérer params depuis cfg_frozen

            # Exécuter
            result, msg = safe_run_backtest(
                engine,
                df,
                cfg_frozen["strategy_key"],
                params,
                cfg_frozen["symbol"],
                cfg_frozen["timeframe"],
                silent_mode=False
            )

            bt_duration = time.time() - start_bt

            if result is None:
                show_status("error", f"Backtest échoué: {msg}")
                return None

            st.success(f"✅ Backtest terminé ({bt_duration:.2f}s)")

            return {
                "result": result,
                "df": df,
                "config": cfg_frozen,
                "duration": bt_duration,
            }

    except Exception as e:
        import traceback
        show_status("error", f"Erreur exécution: {e}")
        st.code(traceback.format_exc())
        return None


def render_backtest_results(backtest_output: Dict[str, Any]) -> None:
    """
    Affiche les résultats du backtest.

    Args:
        backtest_output: Sortie de execute_backtest_with_frozen_config
    """
    result = backtest_output["result"]
    df = backtest_output["df"]
    config = backtest_output["config"]

    st.header("📊 Résultats du Backtest")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = result.get("total_return", 0)
        st.metric("Rendement Total", f"{total_return:.2f}%")

    with col2:
        sharpe = result.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col3:
        max_dd = result.get("max_drawdown", 0)
        st.metric("Max Drawdown", f"{max_dd:.2f}%")

    with col4:
        n_trades = result.get("total_trades", 0)
        st.metric("Trades", f"{n_trades}")

    # Détails configuration
    with st.expander("📋 Configuration utilisée", expanded=False):
        st.json(config)

    # TODO: Ajouter graphiques, liste des trades, etc.
    st.info("🚧 Graphiques et détails à venir")


def main():
    """
    Point d'entrée principal de l'application.
    """
    st.set_page_config(
        page_title="Backtest Engine - Pattern Form",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Backtest Engine v2 - Pattern st.form()")

    # === SIDEBAR: CONFIGURATION ===
    config_validated = render_minimal_config_form()

    # === MAIN: EXÉCUTION ===
    st.header("Configuration & Exécution")

    # Afficher statut configuration
    if config_validated:
        cfg = st.session_state.get("cfg_draft", {})
        st.success(
            f"✅ Configuration validée: {cfg.get('strategy_key')} | "
            f"{cfg.get('symbol')} | {cfg.get('timeframe')}"
        )
    else:
        st.info("💡 Configurez et validez les paramètres dans la sidebar →")
        st.stop()

    # Bouton Run
    run_clicked = render_run_button()

    # Exécution si Run cliqué
    if run_clicked:
        # Récupérer config figée
        cfg_frozen = get_frozen_config()

        if cfg_frozen is None:
            show_status("error", "Configuration non validée")
            st.stop()

        st.info(f"🔒 Configuration figée pour exécution")

        # Exécuter backtest
        backtest_output = execute_backtest_with_frozen_config(cfg_frozen)

        if backtest_output is not None:
            # Stocker résultats dans session_state
            st.session_state["last_backtest_output"] = backtest_output

            # Afficher résultats
            render_backtest_results(backtest_output)

            # Reset validation (optionnel - permet de re-valider pour nouveau run)
            # reset_validation()

    # Afficher derniers résultats si disponibles
    elif "last_backtest_output" in st.session_state:
        st.info("📊 Derniers résultats disponibles")
        render_backtest_results(st.session_state["last_backtest_output"])


if __name__ == "__main__":
    main()