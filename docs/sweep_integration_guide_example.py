"""
Module-ID: docs.sweep_integration_guide_example

Purpose: Exemples d'intégration SweepMonitor dans boucles optimisation - NOT IMPORTED BY RUNTIME.

Role in pipeline: documentation

Key components: Exemples code, patterns, best practices

Inputs: Param grids, données OHLCV

Outputs: Code exemple utilisable

Dependencies: ui.components.sweep_monitor, backtest.engine

Conventions: Fichier de documentation pure (NO RUNTIME IMPORT)

Read-if: Intégrer SweepMonitor dans workflows optimisation custom.

Skip-if: Vous utilisez juste l'UI standard.
"""

from typing import Any, Dict, List

import streamlit as st
import pandas as pd

from ui.components.sweep_monitor import (
    SweepMonitor,
    render_sweep_progress,
    render_sweep_summary,
)


def example_basic_sweep_with_monitor(
    param_grid: List[Dict[str, Any]],
    strategy_name: str,
    data: pd.DataFrame,
) -> SweepMonitor:
    """
    Exemple basique : Boucle d'optimisation avec monitoring temps réel.

    Args:
        param_grid: Liste des combinaisons de paramètres à tester
        strategy_name: Nom de la stratégie
        data: DataFrame OHLCV

    Returns:
        SweepMonitor avec tous les résultats

    Example d'utilisation dans ui/app.py:
        >>> param_grid = generate_param_grid(...)
        >>> monitor = example_basic_sweep_with_monitor(param_grid, "ema_cross", df)
        >>> # Résultats finaux disponibles dans monitor.results
    """
    from backtest.engine import BacktestEngine

    # 1. Créer le moniteur
    monitor = SweepMonitor(
        total_combinations=len(param_grid),
        objectives=["sharpe_ratio", "total_return_pct", "max_drawdown"],
        top_k=10,
    )
    monitor.start()

    # 2. Créer un placeholder Streamlit pour mise à jour dynamique
    progress_placeholder = st.empty()

    # 3. Boucle d'optimisation
    engine = BacktestEngine()

    for i, params in enumerate(param_grid):
        # Exécuter le backtest
        try:
            result = engine.run(
                df=data,
                strategy=strategy_name,
                params=params,
            )
            metrics = result.metrics

            # Mettre à jour le moniteur
            monitor.update(
                params=params,
                metrics=metrics,
                duration_ms=result.execution_time_ms if hasattr(result, "execution_time_ms") else 0.0,
            )

        except Exception as e:
            # En cas d'erreur, marquer comme erreur
            monitor.update(
                params=params,
                metrics={},
                error=True,
            )
            st.warning(f"❌ Erreur pour params {params}: {e}")

        # 4. Mise à jour visuelle en temps réel (toutes les 5 itérations)
        if i % 5 == 0 or i == len(param_grid) - 1:
            with progress_placeholder.container():
                render_sweep_progress(
                    monitor,
                    key=f"sweep_progress_{i}",
                    show_top_results=True,
                    show_evolution=True,
                )

    # 5. Afficher le résumé final
    st.success(f"✅ Optimisation terminée - {monitor.stats.evaluated} combinaisons évaluées")
    render_sweep_summary(monitor, key="sweep_final")

    return monitor


def example_advanced_sweep_with_pruning(
    param_grid: List[Dict[str, Any]],
    strategy_name: str,
    data: pd.DataFrame,
    min_sharpe: float = 0.5,
) -> SweepMonitor:
    """
    Exemple avancé : Optimisation avec pruning (early stopping).

    Args:
        param_grid: Liste des combinaisons
        strategy_name: Nom de la stratégie
        data: DataFrame OHLCV
        min_sharpe: Seuil minimum de Sharpe pour continuer

    Returns:
        SweepMonitor avec résultats

    Example:
        >>> # Arrête les combinaisons similaires si Sharpe < 0.5
        >>> monitor = example_advanced_sweep_with_pruning(param_grid, "ema_cross", df)
    """
    from backtest.engine import BacktestEngine

    monitor = SweepMonitor(
        total_combinations=len(param_grid),
        objectives=["sharpe_ratio", "total_return_pct"],
    )
    monitor.start()

    progress_placeholder = st.empty()
    engine = BacktestEngine()

    for i, params in enumerate(param_grid):
        # Vérifier si on doit prune cette combinaison
        should_prune = False

        # Exemple de critère de pruning : si les 10 derniers résultats sont mauvais
        recent_results = monitor.results[-10:] if len(monitor.results) >= 10 else []
        if recent_results:
            recent_sharpes = [r.sharpe for r in recent_results]
            avg_recent_sharpe = sum(recent_sharpes) / len(recent_sharpes)

            if avg_recent_sharpe < min_sharpe:
                should_prune = True

        if should_prune:
            monitor.update(params=params, metrics={}, pruned=True)
            continue

        # Exécuter normalement
        try:
            result = engine.run(
                df=data,
                strategy=strategy_name,
                params=params,
            )
            monitor.update(params=params, metrics=result.metrics)

        except Exception:
            monitor.update(params=params, metrics={}, error=True)

        # Mise à jour UI
        if i % 3 == 0:
            with progress_placeholder.container():
                render_sweep_progress(monitor, key=f"sweep_{i}")

    render_sweep_summary(monitor, key="sweep_final_pruned")
    return monitor


def example_integration_in_ui_app():
    """
    Exemple d'intégration complète dans ui/app.py.

    COPIER CE CODE dans ui/app.py à l'endroit où vous gérez l'optimisation.
    """
    # === DANS LA SECTION OPTIMISATION DE ui/app.py ===

    # Générer la grille de paramètres
    # param_grid = generate_param_grid(strategy_specs, constraints)

    # Créer et démarrer le moniteur
    # monitor = SweepMonitor(
    #     total_combinations=len(param_grid),
    #     objectives=["sharpe_ratio", "total_return_pct", "max_drawdown"],
    #     top_k=15,
    # )
    # monitor.start()

    # Placeholder pour mise à jour dynamique
    # progress_container = st.empty()

    # Boucle d'optimisation
    # for i, params in enumerate(param_grid):
    #     # Vérifier si l'utilisateur a demandé l'arrêt
    #     if st.session_state.get("stop_requested", False):
    #         st.warning("⛔ Arrêt demandé par l'utilisateur")
    #         break
    #
    #     # Exécuter le backtest
    #     try:
    #         result = engine.run(
    #             df=data,
    #             strategy=strategy_name,
    #             params=params,
    #         )
    #         monitor.update(params, result.metrics, duration_ms=result.execution_time_ms)
    #     except Exception as e:
    #         monitor.update(params, {}, error=True)
    #
    #     # Mise à jour UI tous les 5 runs
    #     if i % 5 == 0 or i == len(param_grid) - 1:
    #         with progress_container.container():
    #             render_sweep_progress(monitor, key=f"opt_progress_{i}")

    # Résumé final
    # st.success(f"✅ {monitor.stats.evaluated} combinaisons évaluées")
    # render_sweep_summary(monitor, key="optimization_summary")

    # Récupérer les meilleurs résultats
    # best_sharpe = monitor.get_best_result("sharpe_ratio")
    # if best_sharpe:
    #     st.success(f"🏆 Meilleur Sharpe: {best_sharpe.sharpe:.3f}")
    #     st.json(best_sharpe.params)

    pass  # Placeholder pour exemple


def streamlit_app_with_sweep_button():
    """
    Exemple complet : Interface Streamlit avec bouton d'optimisation.

    Utilisez ce code comme template pour intégrer le sweep monitor.
    """
    st.title("🔬 Optimisation avec Monitoring Temps Réel")

    # Sidebar : configuration
    with st.sidebar:
        st.header("Configuration")
        strategy_name = st.selectbox("Stratégie", ["ema_cross", "macd_cross", "rsi_reversal"])

        # Plage de paramètres (exemple)
        if strategy_name == "ema_cross":
            fast_min = st.number_input("Fast min", value=5, min_value=2, max_value=50)
            fast_max = st.number_input("Fast max", value=20, min_value=2, max_value=50)
            slow_min = st.number_input("Slow min", value=20, min_value=10, max_value=100)
            slow_max = st.number_input("Slow max", value=50, min_value=10, max_value=100)

    # Bouton de lancement
    if st.button("🚀 Lancer Optimisation", type="primary"):
        # Générer param_grid (exemple simplifié)
        param_grid = [
            {"fast_period": fast, "slow_period": slow}
            for fast in range(fast_min, fast_max + 1, 2)
            for slow in range(slow_min, slow_max + 1, 5)
            if fast < slow
        ]

        st.info(f"📊 {len(param_grid)} combinaisons à tester")

        # Charger les données (exemple)
        # data = load_ohlcv("BTC-USD", "1h")

        # Lancer l'optimisation avec monitoring
        # monitor = example_basic_sweep_with_monitor(param_grid, strategy_name, data)

        # Afficher les meilleurs résultats
        # st.subheader("🏆 Top 5 Résultats")
        # for i, result in enumerate(monitor.get_top_results("sharpe_ratio")[:5], 1):
        #     st.write(f"{i}. Sharpe: {result.sharpe:.3f} | Params: {result.params}")


# === AIDE RAPIDE ===

"""
RÉSUMÉ RAPIDE - Comment intégrer SweepMonitor :

1. CRÉER le moniteur AVANT la boucle :
   monitor = SweepMonitor(total_combinations=len(param_grid))
   monitor.start()

2. CRÉER un placeholder Streamlit :
   progress_placeholder = st.empty()

3. DANS LA BOUCLE, après chaque backtest :
   monitor.update(params=params, metrics=result.metrics)

4. MISE À JOUR UI (toutes les N itérations) :
   with progress_placeholder.container():
       render_sweep_progress(monitor, key=f"progress_{i}")

5. APRÈS LA BOUCLE, résumé final :
   render_sweep_summary(monitor, key="final")

6. RÉCUPÉRER les meilleurs résultats :
   best = monitor.get_best_result("sharpe_ratio")
   st.json(best.params)
"""

__all__ = [
    "example_basic_sweep_with_monitor",
    "example_advanced_sweep_with_pruning",
    "example_integration_in_ui_app",
    "streamlit_app_with_sweep_button",
]
