"""
Module-ID: ui.results

Purpose: Affiche les résultats détaillés des backtests avec métriques et graphiques.

Role in pipeline: reporting

Key components: render_results, métriques, graphiques

Inputs: SidebarState, résultats de backtest

Outputs: Interface Streamlit avec métriques et visualisations

Dependencies: ui.state, ui.components.charts

Conventions: Métriques financières standardisées

Read-if: Affichage des résultats de backtest

Skip-if: Pas de résultats à afficher
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from ui.components.charts import (
    render_equity_and_drawdown,
    render_ohlcv_with_trades,
    render_ohlcv_with_trades_and_indicators,
    render_returns_distribution,
    render_trade_pnl_distribution,
)
from ui.context import resolve_latest_version, save_versioned_preset
from ui.helpers import build_indicator_overlays, generate_strategies_table
from ui.log_taps import BestPnlTracker
from ui.state import SidebarState


def render_results(state: SidebarState, best_pnl_tracker: Optional[BestPnlTracker]) -> None:
    result = st.session_state.get("last_run_result")
    winner_params = st.session_state.get("last_winner_params")
    winner_metrics = st.session_state.get("last_winner_metrics")
    winner_origin = st.session_state.get("last_winner_origin")
    winner_meta = st.session_state.get("last_winner_meta")

    if result is not None:
        st.header("📊 Résultats du Backtest")

        col1, col2, col3, col4, col5 = st.columns(5)

        if result is not None:
            with col1:
                pnl = result.metrics["total_pnl"]
                pnl_color: str = "normal" if pnl >= 0 else "inverse"
                ret_pct = result.metrics["total_return_pct"]
                st.metric(
                    "P&L Total",
                    f"${pnl:,.2f}",
                    delta=f"{ret_pct:.1f}%",
                    delta_color=pnl_color,  # type: ignore[arg-type]
                )

            with col2:
                sharpe = result.metrics["sharpe_ratio"]
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with col3:
                # Compatibilité: engine restauré retourne "max_drawdown", UI récent cherche "max_drawdown_pct"
                max_dd = result.metrics.get("max_drawdown_pct", result.metrics.get("max_drawdown", 0))
                st.metric("Max Drawdown", f"{max_dd:.1f}%")

            with col4:
                trades = result.metrics["total_trades"]
                # Compatibilité: engine restauré retourne "win_rate", UI récent cherche "win_rate_pct"
                win_rate = result.metrics.get("win_rate_pct", result.metrics.get("win_rate", 0))
                st.metric("Trades", f"{trades}", delta=f"{win_rate:.0f}% wins")

            with col5:
                if best_pnl_tracker is None:
                    st.metric("Backtest PnL (best run)", "n/a")
                else:
                    best_pnl, best_run_id = best_pnl_tracker.get_best()
                    if best_pnl is None:
                        st.metric("Backtest PnL (best run)", "n/a")
                    else:
                        st.metric(
                            "Backtest PnL (best run)",
                            f"${best_pnl:,.2f}",
                        )
                        if best_run_id:
                            st.caption(f"run {best_run_id}")

        liquidation_pnl = result.metrics.get("liquidation_total_pnl")
        if liquidation_pnl is not None:
            if result.metrics.get("liquidation_triggered"):
                liquidation_time = result.metrics.get("liquidation_time")
                time_note = f" à {liquidation_time}" if liquidation_time else ""
                st.warning(
                    f"💥 Liquidation détectée{time_note}. "
                    "Le mode liquidation coupe les trades dès que le capital atteint 0."
                )

            with st.expander("🧯 Liquidation vs crédit infini", expanded=False):
                credit_col, liq_col = st.columns(2)
                with credit_col:
                    st.markdown("**Crédit infini**")
                    st.metric("P&L", f"${result.metrics.get('total_pnl', 0):,.2f}")
                    st.metric("Sharpe", f"{result.metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Max DD", f"{result.metrics.get('max_drawdown_pct', 0):.1f}%")
                    st.metric("Trades", f"{result.metrics.get('total_trades', 0)}")
                with liq_col:
                    st.markdown("**Liquidation**")
                    st.metric("P&L", f"${result.metrics.get('liquidation_total_pnl', 0):,.2f}")
                    st.metric(
                        "Sharpe",
                        f"{result.metrics.get('liquidation_sharpe_ratio', 0):.2f}",
                    )
                    st.metric(
                        "Max DD",
                        f"{result.metrics.get('liquidation_max_drawdown_pct', 0):.1f}%",
                    )
                    st.metric(
                        "Trades",
                        f"{result.metrics.get('liquidation_total_trades', result.metrics.get('total_trades', 0))}",
                    )

        if result is not None and winner_params is not None:
            st.subheader("Versioned preset")
            col_save_a, col_save_b = st.columns(2)

            with col_save_a:
                default_version = resolve_latest_version(state.strategy_key)
                preset_version = st.text_input(
                    "Preset version",
                    value=default_version,
                    key="winner_preset_version",
                )
                preset_name = st.text_input(
                    "Preset name",
                    value="winner",
                    key="winner_preset_name",
                )

            with col_save_b:
                description_default = (
                    f"{state.strategy_key} winner {state.symbol}/{state.timeframe}"
                )
                preset_description = st.text_input(
                    "Description",
                    value=description_default,
                    key="winner_preset_description",
                )

            if st.button("Save winner preset", key="save_winner_preset"):
                extra_meta = {}
                if winner_meta:
                    for key in [
                        "symbol",
                        "timeframe",
                        "period_start",
                        "period_end",
                    ]:
                        if key in winner_meta:
                            extra_meta[key] = winner_meta[key]

                origin_run_id = None
                if winner_meta and "run_id" in winner_meta:
                    origin_run_id = winner_meta["run_id"]

                try:
                    saved = save_versioned_preset(
                        strategy_name=state.strategy_key,
                        version=preset_version,
                        preset_name=preset_name,
                        params_values=winner_params,
                        indicators=state.strategy_info.required_indicators
                        if state.strategy_info is not None
                        else None,
                        description=preset_description,
                        metrics=winner_metrics,
                        origin=winner_origin,
                        origin_run_id=origin_run_id,
                        extra_metadata=extra_meta,
                    )
                    st.session_state["_sync_preset_version"] = preset_version
                    st.session_state["_sync_preset_name"] = saved.name
                    st.session_state["versioned_preset_last_saved"] = saved.name
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

        st.subheader("💰 Courbe d'Équité")

        if result is not None and hasattr(result, "equity") and result.equity is not None:
            initial_capital = state.params.get("initial_capital", 10000.0)
            render_equity_and_drawdown(
                equity=result.equity,
                initial_capital=initial_capital,
                key="equity_drawdown_main",
                height=550,
            )
        elif result is not None:
            st.info("Courbe d'équité non disponible pour cette stratégie")

        st.subheader("📈 Prix et Trades")

        if result is not None:
            chart_df = st.session_state.get("ohlcv_df")
            if chart_df is None:
                st.info("Donnees non chargees. Cliquez sur 'Charger donnees'.")
            else:
                chart_params = result.meta.get("params", state.params)
                indicator_overlays = build_indicator_overlays(
                    state.strategy_key, chart_df, chart_params
                )

                if indicator_overlays:
                    render_ohlcv_with_trades_and_indicators(
                        df=chart_df,
                        trades_df=result.trades,
                        overlays=indicator_overlays,
                        active_indicators=state.active_indicators,
                        title="📊 OHLCV + Indicateurs + Entrees/Sorties",
                        key="ohlcv_trades_indicators_main",
                        height=700,
                    )
                elif not result.trades.empty:
                    render_ohlcv_with_trades(
                        df=chart_df,
                        trades_df=result.trades,
                        title="📊 Graphique OHLCV avec Points d'Entree/Sortie",
                        key="ohlcv_trades_main",
                        height=600,
                    )
                else:
                    st.info(
                        "Aucun trade execute, affichage du graphique de prix uniquement"
                    )
                    render_ohlcv_with_trades(
                        df=chart_df,
                        trades_df=pd.DataFrame(),
                        title="📊 Graphique OHLCV",
                        key="ohlcv_main_notrades",
                        height=600,
                    )

        st.subheader("📈 Métriques Détaillées")

        if result is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**💰 Rendement**")
                st.text(f"P&L Total: ${result.metrics['total_pnl']:,.2f}")
                st.text(f"Rendement: {result.metrics['total_return_pct']:.2f}%")
                st.text(f"Ann. Return: {result.metrics['annualized_return']:.2f}%")
                st.text(f"Volatilité: {result.metrics['volatility_annual']:.2f}%")

            with col2:
                st.markdown("**📊 Risque**")
                st.text(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
                st.text(f"Sortino: {result.metrics['sortino_ratio']:.2f}")
                st.text(f"Calmar: {result.metrics['calmar_ratio']:.2f}")
                # Compatibilité: fallback max_drawdown si max_drawdown_pct absent
                max_dd = result.metrics.get('max_drawdown_pct', result.metrics.get('max_drawdown', 0))
                st.text(f"Max DD: {max_dd:.2f}%")

            with col3:
                st.markdown("**🎯 Trading**")
                st.text(f"Trades: {result.metrics['total_trades']}")
                # Compatibilité: fallback win_rate si win_rate_pct absent
                win_rate = result.metrics.get('win_rate_pct', result.metrics.get('win_rate', 0))
                st.text(f"Win Rate: {win_rate:.1f}%")
                st.text(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
                st.text(f"Expectancy: ${result.metrics['expectancy']:.2f}")

        if result is not None and not result.trades.empty:
            with st.expander("📊 Analyse Statistique Avancée (Seaborn)", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    render_trade_pnl_distribution(
                        trades_df=result.trades,
                        title="Distribution des P&L par Trade",
                        key="pnl_dist_main",
                        height=400,
                    )

                with col2:
                    if hasattr(result, "returns") and result.returns is not None:
                        render_returns_distribution(
                            returns=result.returns,
                            title="Distribution des Rendements",
                            key="returns_dist_main",
                            height=400,
                        )
                    else:
                        st.info("Rendements non disponibles pour cette analyse")

        if result is not None and not result.trades.empty:
            st.subheader("📋 Historique des Trades")

            trades_display = result.trades.copy()

            if "entry_ts" in trades_display.columns:
                trades_display["entry_ts"] = pd.to_datetime(
                    trades_display["entry_ts"]
                ).dt.strftime("%Y-%m-%d %H:%M")
            if "exit_ts" in trades_display.columns:
                trades_display["exit_ts"] = pd.to_datetime(
                    trades_display["exit_ts"]
                ).dt.strftime("%Y-%m-%d %H:%M")
            if "pnl" in trades_display.columns:
                trades_display["pnl"] = trades_display["pnl"].apply(
                    lambda x: f"${x:,.2f}"
                )
            if "price_entry" in trades_display.columns:
                trades_display["price_entry"] = trades_display[
                    "price_entry"
                ].apply(lambda x: f"${x:,.2f}")
            if "price_exit" in trades_display.columns:
                trades_display["price_exit"] = trades_display[
                    "price_exit"
                ].apply(lambda x: f"${x:,.2f}")

            cols_to_show = [
                "entry_ts",
                "exit_ts",
                "side",
                "price_entry",
                "price_exit",
                "pnl",
                "return_pct",
                "exit_reason",
            ]
            display_cols = [c for c in cols_to_show if c in trades_display.columns]

            st.dataframe(trades_display[display_cols], width="stretch")

            total_trades = len(result.trades)
            winners = (result.trades["pnl"] > 0).sum()
            losers = (result.trades["pnl"] < 0).sum()
            st.caption(
                f"Total: {total_trades} | Gagnants: {winners} | Perdants: {losers}"
            )
        elif result is not None:
            st.info("Aucun trade exécuté pendant cette période")

    else:
        render_home(state)


def render_home(state: SidebarState) -> None:
    st.info("👆 Configurez dans la sidebar puis cliquez sur **🚀 Lancer le Backtest**")

    llm_mode_active = state.optimization_mode == "🤖 Optimisation LLM"

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Stratégies", "📊 Optimisation", "📁 Données", "❓ FAQ"]
    )

    with tab1:
        strategies_table = generate_strategies_table()
        st.markdown(strategies_table)

        st.markdown(
            """
        ### Indicateurs Intégrés
        - Bollinger Bands, ATR, RSI, EMA, SMA, MACD, ADX
        - Ichimoku, PSAR, Stochastic RSI, Vortex, etc.
        """
        )

    with tab2:
        st.markdown(
            """
        ### Système d'Optimisation

        **Mode Grille** *(par défaut)* : Test de multiples combinaisons.
        - Définissez Min/Max/Step pour chaque paramètre
        - Le système calcule toutes les combinaisons
        - Limite configurable (jusqu'à 1,000,000)

        **Mode Simple** : Test d'une seule combinaison de paramètres.
        """
        )

        table_lines = [
            "| Mode | Combinaisons | Intelligence | Coût |",
            "|------|--------------|--------------|------|",
            "| Simple | 1 | ❌ | Gratuit |",
            "| Grille | Jusqu'à 1M | ❌ | Gratuit |",
        ]
        if llm_mode_active:
            table_lines.append("| LLM | ~10-50 ciblées | ✅ | Variable |")
        st.markdown("\n".join(table_lines))

        if llm_mode_active:
            st.markdown(
                """
        **Mode LLM** 🤖 : Optimisation intelligente par agents IA.
        - 4 agents spécialisés (Analyst, Strategist, Critic, Validator)
        - Boucle d'amélioration itérative automatique
        - Walk-Forward anti-overfitting intégré
        - Supporte Ollama (local/gratuit) ou OpenAI

        ⚠️ Mode LLM nécessite Ollama installé localement ou une clé OpenAI.
        """
            )

    with tab3:
        st.markdown(
            f"""
        ### Format des Données

        Les données OHLCV doivent être au format Parquet ou CSV:
        - `SYMBOL_TIMEFRAME.parquet` (ex: `BTCUSDT_1h.parquet`)

        **Symboles détectés**: {len(state.available_tokens)}
        **Timeframes**: {', '.join(state.available_timeframes)}
        """
        )

    with tab4:
        st.markdown(
            """
        ### Questions Fréquentes

        **Q: Comment tester plus de combinaisons?**
        R: En mode Grille, définissez Min/Max/Step pour chaque paramètre.
        Augmentez la limite de combinaisons si nécessaire.

        **Q: Que signifie le Sharpe Ratio?**
        R: Rendement ajusté au risque. > 1 = bon, > 2 = excellent.

        **Q: Pourquoi le mode Grille est lent?**
        R: Il teste toutes les combinaisons. Augmentez le Step ou réduisez la plage.
        """
        )

        if llm_mode_active:
            st.markdown(
                """

        **Q: Comment éviter l'overfitting?**
        R: Utilisez le Walk-Forward Validation (activé par défaut en mode LLM).

        **Q: Comment fonctionne le mode LLM?**
        R: 4 agents IA travaillent ensemble:
        1. **Analyst** analyse les métriques actuelles
        2. **Strategist** propose de nouveaux paramètres
        3. **Critic** détecte l'overfitting potentiel
        4. **Validator** décide: approuver, rejeter ou itérer

        **Q: Ollama vs OpenAI?**
        R: Ollama est gratuit et local (installer depuis ollama.ai).
        OpenAI est plus puissant mais payant (~0.01$/requête).
        """
            )
