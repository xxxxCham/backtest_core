"""
Backtest Core - Chart Components
=================================

Composants de graphiques réutilisables avec Plotly.

Features:
- Courbe d'équité + Drawdown
- Prix OHLCV avec indicateurs
- Marqueurs de trades (entrée/sortie)
- Graphiques de comparaison de résultats
- Style moderne et cohérent

Usage:
    >>> from ui.components.charts import render_equity_and_drawdown
    >>> render_equity_and_drawdown(equity, drawdown)
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.log import get_logger

logger = get_logger(__name__)


def render_equity_and_drawdown(
    equity: pd.Series,
    initial_capital: float = 10000.0,
    key: str = "equity_dd",
    height: int = 550,
) -> None:
    """
    Affiche la courbe d'équité et le drawdown dans un graphique à 2 panneaux.

    Args:
        equity: Série pandas de l'équité
        initial_capital: Capital initial
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if equity is None or equity.empty:
        st.warning("⚠️ Aucune donnée d'équité à afficher")
        return

    # Calculer le drawdown
    cummax = equity.expanding().max()
    drawdown = ((equity - cummax) / cummax) * 100

    # Créer le graphique à 2 sous-graphiques
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("💰 Équité ($)", "📉 Drawdown (%)"),
    )

    # Graphique d'équité
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Équité",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.15)",
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Ligne du capital initial
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="rgba(200, 200, 200, 0.5)",
        annotation_text=f"Capital: ${initial_capital:,.0f}",
        annotation_position="left",
        row=1,
        col=1,
    )

    # Graphique de drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ef5350", width=1),
            fillcolor="rgba(239, 83, 80, 0.3)",
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Mise en forme
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=40, b=30),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8b2d1", size=11),
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(title_text="$", gridcolor="rgba(128,128,128,0.1)", row=1, col=1)
    fig.update_yaxes(title_text="%", gridcolor="rgba(128,128,128,0.1)", row=2, col=1)

    st.plotly_chart(fig, width='stretch', key=key)


def render_ohlcv_with_trades(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    title: str = "📈 Prix et Trades",
    key: str = "ohlcv_trades",
    height: int = 500,
) -> None:
    """
    Affiche un graphique OHLCV avec marqueurs de trades.

    Args:
        df: DataFrame OHLCV avec colonnes open, high, low, close
        trades_df: DataFrame des trades avec entry_time, exit_time, entry_price, exit_price
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if df.empty:
        st.warning("⚠️ Aucune donnée OHLCV")
        return

    # Vérifier les colonnes requises
    required_ohlc = {"open", "high", "low", "close"}
    if not required_ohlc.issubset(set(df.columns)):
        st.error(f"❌ Colonnes manquantes: {required_ohlc - set(df.columns)}")
        return

    st.markdown(f"#### {title}")

    fig = go.Figure()

    # Candlestick OHLC
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Ajouter les marqueurs de trades
    if not trades_df.empty and "entry_time" in trades_df.columns:
        # Points d'entrée
        entries = trades_df[trades_df["entry_time"].notna()].copy()
        if not entries.empty:
            # Déterminer les couleurs par type de trade (LONG/SHORT)
            entry_colors = entries.get("side", "LONG").map(
                {"LONG": "#42a5f5", "SHORT": "#ab47bc"}
            ).fillna("#42a5f5")

            fig.add_trace(
                go.Scatter(
                    x=entries["entry_time"],
                    y=entries["entry_price"],
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color=entry_colors,
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="<b>Entry</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
                )
            )

        # Points de sortie
        exits = trades_df[trades_df["exit_time"].notna()].copy()
        if not exits.empty:
            # Couleurs basées sur profit/loss
            exit_colors = []
            pnl_values = []
            for _, trade in exits.iterrows():
                pnl = trade.get("pnl", 0)
                pnl_values.append(pnl)
                exit_colors.append("#4caf50" if pnl > 0 else "#f44336")

            fig.add_trace(
                go.Scatter(
                    x=exits["exit_time"],
                    y=exits["exit_price"],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color=exit_colors,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=pnl_values,
                    hovertemplate="<b>Exit</b><br>%{x}<br>Prix: $%{y:.2f}<br>PNL: $%{customdata:.2f}<extra></extra>",
                )
            )

    _apply_chart_layout(fig, height=height, y_title="Prix (USD)")
    st.plotly_chart(fig, width='stretch', key=key)


def render_ohlcv_with_indicators(
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    title: str = "📊 Prix et Indicateurs",
    key: str = "ohlcv_indicators",
    height: int = 500,
) -> None:
    """
    Affiche un graphique OHLCV avec indicateurs techniques.

    Args:
        df: DataFrame OHLCV
        indicators: Dict d'indicateurs (ex: {'bollinger': {'upper': series, 'lower': series}})
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if df.empty:
        st.warning("⚠️ Aucune donnée OHLCV")
        return

    st.markdown(f"#### {title}")

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Ajouter les indicateurs
    if "bollinger" in indicators:
        bb = indicators["bollinger"]
        if "upper" in bb and "lower" in bb and "mid" in bb:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bb["mid"],
                    name="BB Mid",
                    line=dict(color="#ffa726", width=1),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bb["upper"],
                    name="BB Upper",
                    line=dict(color="#42a5f5", width=1, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bb["lower"],
                    name="BB Lower",
                    line=dict(color="#42a5f5", width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(66, 165, 245, 0.1)",
                )
            )

    _apply_chart_layout(fig, height=height, y_title="Prix (USD)")
    st.plotly_chart(fig, width='stretch', key=key)


def render_equity_curve(
    equity: pd.Series,
    initial_capital: float = 10000.0,
    title: str = "💹 Courbe d'Équité",
    key: str = "equity_curve",
    height: int = 350,
) -> None:
    """
    Affiche uniquement la courbe d'équité (version simple).

    Args:
        equity: Série pandas de l'équité
        initial_capital: Capital initial
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if equity is None or equity.empty:
        st.warning("⚠️ Aucune donnée d'équité")
        return

    st.markdown(f"#### {title}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Équité",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.15)",
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
        )
    )

    # Ligne du capital initial
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="rgba(200, 200, 200, 0.5)",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
        annotation_position="right",
    )

    _apply_chart_layout(fig, height=height, y_title="Équité ($)")
    st.plotly_chart(fig, width='stretch', key=key)


def render_comparison_chart(
    results_list: List[Dict[str, Any]],
    metric: str = "sharpe_ratio",
    title: str = "📊 Comparaison des Résultats",
    key: str = "comparison",
    height: int = 400,
) -> None:
    """
    Affiche un graphique de comparaison entre plusieurs résultats.

    Args:
        results_list: Liste de dict avec 'name' et 'metrics'
        metric: Métrique à comparer
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if not results_list:
        st.warning("⚠️ Aucun résultat à comparer")
        return

    st.markdown(f"#### {title}")

    names = [r.get("name", f"Run {i}") for i, r in enumerate(results_list)]
    values = [r.get("metrics", {}).get(metric, 0) for r in results_list]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker_color=["#26a69a" if v > 0 else "#ef5350" for v in values],
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=metric.replace("_", " ").title(),
        height=height,
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8b2d1", size=11),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    st.plotly_chart(fig, width='stretch', key=key)


def _apply_chart_layout(
    fig: go.Figure,
    height: int = 450,
    y_title: str = "",
    show_rangeslider: bool = False,
) -> None:
    """
    Applique un style cohérent aux graphiques Plotly.

    Args:
        fig: Figure Plotly
        height: Hauteur du graphique
        y_title: Titre de l'axe Y
        show_rangeslider: Afficher le range slider en bas
    """
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=50, t=30, b=30),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title=y_title,
        xaxis=dict(
            rangeslider=dict(visible=show_rangeslider),
            gridcolor="rgba(128,128,128,0.1)",
        ),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8b2d1", size=11),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )


__all__ = [
    "render_equity_and_drawdown",
    "render_ohlcv_with_trades",
    "render_ohlcv_with_indicators",
    "render_equity_curve",
    "render_comparison_chart",
]
