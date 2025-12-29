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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.log import get_logger

# Import optionnel de seaborn pour distributions statistiques
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import optionnel de plotly-resampler pour downsampling intelligent
try:
    from plotly_resampler import FigureResampler
    PLOTLY_RESAMPLER_AVAILABLE = True
except ImportError:
    PLOTLY_RESAMPLER_AVAILABLE = False

logger = get_logger(__name__)

PLOTLY_CHART_CONFIG = {"scrollZoom": True}
RESAMPLER_THRESHOLD = 100000  # Utiliser resampler si > 100k points

def _wrap_with_resampler(fig: go.Figure, n_datapoints: int) -> go.Figure:
    """
    Wrap une figure Plotly avec FigureResampler si le dataset est grand.

    Args:
        fig: Figure Plotly originale
        n_datapoints: Nombre de points de données

    Returns:
        Figure Plotly (wrappée ou non)
    """
    if PLOTLY_RESAMPLER_AVAILABLE and n_datapoints > RESAMPLER_THRESHOLD:
        logger.info(f"Dataset large ({n_datapoints:,} points) - Activation du resampler")
        try:
            # Convertir en FigureResampler pour downsampling intelligent
            return FigureResampler(fig, default_n_shown_samples=2000)
        except Exception as e:
            logger.warning(f"Échec du resampler: {e} - Utilisation de la figure standard")
            return fig
    return fig

def _apply_axis_interaction(fig: go.Figure, lock_x: bool = False) -> None:
    """Enable zoom on Y while keeping X interactive when needed."""
    fig.update_layout(dragmode="zoom")
    fig.update_xaxes(fixedrange=lock_x)
    fig.update_yaxes(fixedrange=False)


def _normalize_trades_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade column names for chart rendering."""
    if trades_df is None or trades_df.empty:
        return trades_df

    rename_map = {}
    if "entry_time" not in trades_df.columns and "entry_ts" in trades_df.columns:
        rename_map["entry_ts"] = "entry_time"
    if "exit_time" not in trades_df.columns and "exit_ts" in trades_df.columns:
        rename_map["exit_ts"] = "exit_time"
    if "entry_price" not in trades_df.columns and "price_entry" in trades_df.columns:
        rename_map["price_entry"] = "entry_price"
    if "exit_price" not in trades_df.columns and "price_exit" in trades_df.columns:
        rename_map["price_exit"] = "exit_price"

    if not rename_map:
        return trades_df

    return trades_df.rename(columns=rename_map)


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
    _apply_axis_interaction(fig)

    # Wrapper avec resampler si grand dataset
    fig = _wrap_with_resampler(fig, len(equity))

    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )


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

    trades_df = _normalize_trades_df(trades_df)

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
    _apply_axis_interaction(fig)
    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )

def render_ohlcv_with_trades_and_indicators(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    overlays: Dict[str, Any],
    active_indicators: Optional[List[str]] = None,
    title: str = "📈 Prix, Indicateurs et Trades",
    key: str = "ohlcv_trades_indicators",
    height: int = 650,
) -> None:
    """
    Affiche un graphique OHLCV avec indicateurs et marqueurs de trades.
    """
    if df.empty:
        st.warning("⚠️ Aucune donnée OHLCV")
        return

    if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
        st.error("❌ Colonnes OHLC manquantes")
        return

    trades_df = _normalize_trades_df(trades_df)

    if active_indicators is None:
        active_set = set(overlays.keys())
    else:
        active_set = set(active_indicators)
    active_set = {name for name in active_set if name in overlays}
    has_macd = "macd" in active_set
    has_rsi = "rsi" in active_set
    has_atr = "atr" in active_set
    has_stochastic = "stochastic" in active_set
    show_second_panel = has_macd or has_rsi or has_atr or has_stochastic

    def _add_trace(trace, row: int = 1) -> None:
        if show_second_panel:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    if show_second_panel:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, "Oscillators"),
        )
    else:
        fig = go.Figure()

    candlestick = go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )
    _add_trace(candlestick, row=1)

    if "bollinger" in active_set:
        bb = overlays["bollinger"]
        upper = bb.get("upper")
        lower = bb.get("lower")
        mid = bb.get("mid")
        entry_upper = bb.get("entry_upper")
        entry_lower = bb.get("entry_lower")
        if mid is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=mid,
                    name="BB Mid",
                    line=dict(color="#ffa726", width=1),
                ),
                row=1,
            )
        if upper is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    name="BB Upper",
                    line=dict(color="#42a5f5", width=1, dash="dash"),
                ),
                row=1,
            )
        if lower is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower,
                    name="BB Lower",
                    line=dict(color="#42a5f5", width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(66, 165, 245, 0.1)",
                ),
                row=1,
            )
        if entry_upper is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=entry_upper,
                    name="Entry Z haut",
                    line=dict(color="#ffcc80", width=1, dash="dot"),
                ),
                row=1,
            )
        if entry_lower is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=entry_lower,
                    name="Entry Z bas",
                    line=dict(color="#ffcc80", width=1, dash="dot"),
                ),
                row=1,
            )

    if "ema" in active_set:
        ema = overlays["ema"]
        fast = ema.get("fast")
        slow = ema.get("slow")
        center = ema.get("center")
        if fast is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=fast,
                    name="EMA rapide",
                    line=dict(color="#42a5f5", width=1.4),
                ),
                row=1,
            )
        if slow is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=slow,
                    name="EMA lente",
                    line=dict(color="#ffb74d", width=1.4),
                ),
                row=1,
            )
        if center is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=center,
                    name="EMA centre",
                    line=dict(color="#42a5f5", width=1.4, dash="dot"),
                ),
                row=1,
            )

    if "ma" in active_set:
        ma = overlays["ma"]
        fast = ma.get("fast")
        slow = ma.get("slow")
        center = ma.get("center")
        if fast is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=fast,
                    name="MA rapide",
                    line=dict(color="#42a5f5", width=1.4),
                ),
                row=1,
            )
        if slow is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=slow,
                    name="MA lente",
                    line=dict(color="#ffb74d", width=1.4),
                ),
                row=1,
            )
        if center is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=center,
                    name="MA",
                    line=dict(color="#42a5f5", width=1.4, dash="dot"),
                ),
                row=1,
            )

    if "atr_channel" in active_set:
        channel = overlays["atr_channel"]
        upper = channel.get("upper")
        lower = channel.get("lower")
        center = channel.get("center")
        if upper is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    name="Canal haut",
                    line=dict(color="#ef5350", width=1.2),
                ),
                row=1,
            )
        if lower is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower,
                    name="Canal bas",
                    line=dict(color="#26a69a", width=1.2),
                ),
                row=1,
            )
        if center is not None:
            _add_trace(
                go.Scatter(
                    x=df.index,
                    y=center,
                    name="EMA centre",
                    line=dict(color="#42a5f5", width=1.2, dash="dot"),
                ),
                row=1,
            )

    if not trades_df.empty and "entry_time" in trades_df.columns:
        entries = trades_df[trades_df["entry_time"].notna()].copy()
        if not entries.empty:
            entry_colors = entries.get("side", "LONG").map(
                {"LONG": "#42a5f5", "SHORT": "#ab47bc"}
            ).fillna("#42a5f5")
            _add_trace(
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
                ),
                row=1,
            )

        exits = trades_df[trades_df["exit_time"].notna()].copy()
        if not exits.empty:
            exit_colors = []
            pnl_values = []
            for _, trade in exits.iterrows():
                pnl = trade.get("pnl", 0)
                pnl_values.append(pnl)
                exit_colors.append("#4caf50" if pnl > 0 else "#f44336")

            _add_trace(
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
                ),
                row=1,
            )

    if show_second_panel:
        if has_macd:
            macd = overlays["macd"]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd.get("macd"),
                    name="MACD",
                    line=dict(color="#26a69a", width=1.4),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd.get("signal"),
                    name="Signal",
                    line=dict(color="#ef5350", width=1.2),
                ),
                row=2,
                col=1,
            )
        if has_rsi:
            rsi = overlays["rsi"]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi.get("rsi"),
                    name="RSI",
                    line=dict(color="#42a5f5", width=1.4),
                ),
                row=2,
                col=1,
            )
            oversold = rsi.get("oversold")
            overbought = rsi.get("overbought")
            if oversold is not None:
                fig.add_hline(
                    y=oversold,
                    line_dash="dot",
                    line_color="#26a69a",
                    annotation_text="Oversold",
                    annotation_position="bottom left",
                    row=2,
                    col=1,
                )
            if overbought is not None:
                fig.add_hline(
                    y=overbought,
                    line_dash="dot",
                    line_color="#ef5350",
                    annotation_text="Overbought",
                    annotation_position="top left",
                    row=2,
                    col=1,
                )
        if has_stochastic:
            stoch = overlays["stochastic"]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch.get("k"),
                    name="%K",
                    line=dict(color="#42a5f5", width=1.2),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch.get("d"),
                    name="%D",
                    line=dict(color="#ffb74d", width=1.2),
                ),
                row=2,
                col=1,
            )
            oversold = stoch.get("oversold")
            overbought = stoch.get("overbought")
            if oversold is not None:
                fig.add_hline(
                    y=oversold,
                    line_dash="dot",
                    line_color="#26a69a",
                    annotation_text="Oversold",
                    annotation_position="bottom left",
                    row=2,
                    col=1,
                )
            if overbought is not None:
                fig.add_hline(
                    y=overbought,
                    line_dash="dot",
                    line_color="#ef5350",
                    annotation_text="Overbought",
                    annotation_position="top left",
                    row=2,
                    col=1,
                )
        if has_atr:
            atr = overlays["atr"]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=atr.get("atr"),
                    name="ATR",
                    line=dict(color="#ab47bc", width=1.4),
                ),
                row=2,
                col=1,
            )
            threshold = atr.get("threshold")
            if threshold is not None:
                fig.add_hline(
                    y=threshold,
                    line_dash="dot",
                    line_color="#ffa726",
                    annotation_text="ATR Threshold",
                    annotation_position="top left",
                    row=2,
                    col=1,
                )

        fig.update_layout(
            height=height,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")
        _apply_axis_interaction(fig)

        # Wrapper avec resampler si grand dataset
        fig = _wrap_with_resampler(fig, len(df))

        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        return

    _apply_chart_layout(fig, height=height, y_title="Prix (USD)")
    _apply_axis_interaction(fig)

    # Wrapper avec resampler si grand dataset
    fig = _wrap_with_resampler(fig, len(df))

    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )


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
    _apply_axis_interaction(fig)
    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )


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
    _apply_axis_interaction(fig)
    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )


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

    _apply_axis_interaction(fig)
    st.plotly_chart(
        fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
    )


def render_strategy_param_diagram(
    strategy_key: str,
    params: Dict[str, Any],
    key: str = "strategy_diagram",
) -> None:
    """
    Affiche un schema explicatif des indicateurs et parametres d'une strategie.
    """
    n = 160
    x = np.arange(n)
    base = 100 + 4 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = 0.9 * np.sin(np.linspace(0, 11 * np.pi, n))
    price = base + noise
    price_series = pd.Series(price)

    if strategy_key == "bollinger_atr":
        bb_period = max(2, min(int(params.get("bb_period", 20)), n - 1))
        bb_std = float(params.get("bb_std", 2.0))
        entry_z = float(params.get("entry_z", bb_std))
        atr_period = max(2, min(int(params.get("atr_period", 14)), n - 1))
        atr_percentile = float(params.get("atr_percentile", 30))
        k_sl = float(params.get("k_sl", 1.5))

        middle = price_series.rolling(window=bb_period, min_periods=1).mean()
        sigma = price_series.rolling(window=bb_period, min_periods=1).std(ddof=0).fillna(0.5)
        upper = middle + sigma * bb_std
        lower = middle - sigma * bb_std
        entry_upper = middle + sigma * entry_z
        entry_lower = middle - sigma * entry_z

        atr_values = price_series.diff().abs().rolling(window=atr_period, min_periods=1).mean()
        atr_threshold = float(np.nanpercentile(atr_values, atr_percentile))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                "Prix + Bandes de Bollinger (entry_z)",
                "ATR et filtre de volatilite (atr_percentile)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                name="Bollinger bas",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                name="Bollinger haut",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
                fill="tonexty",
                fillcolor="rgba(100, 160, 200, 0.15)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=middle,
                name="Bollinger milieu",
                line=dict(color="rgba(140, 200, 255, 0.9)", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_upper,
                name="Seuil entry_z haut",
                line=dict(color="rgba(255, 204, 128, 0.9)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_lower,
                name="Seuil entry_z bas",
                line=dict(color="rgba(255, 204, 128, 0.9)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            ),
            row=1,
            col=1,
        )

        entry_index = int(n * 0.72)
        entry_price = price[entry_index]
        atr_at_entry = float(atr_values.iloc[entry_index])
        stop_price = entry_price - k_sl * atr_at_entry
        fig.add_trace(
            go.Scatter(
                x=[entry_index],
                y=[entry_price],
                mode="markers",
                name="Exemple entree",
                marker=dict(color="#26a69a", size=8),
            ),
            row=1,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=entry_index,
            x1=entry_index,
            y0=entry_price,
            y1=stop_price,
            line=dict(color="#ef5350", width=2),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=entry_index,
            y=stop_price,
            text=f"Stop = k_sl x ATR ({k_sl:.2f})",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=20,
            font=dict(color="#ef9a9a", size=10),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=atr_values,
                name="ATR",
                line=dict(color="#ab47bc", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=atr_threshold,
            line_dash="dot",
            line_color="#ffa726",
            annotation_text=f"Seuil {atr_percentile:.0f}%",
            annotation_position="top left",
            row=2,
            col=1,
        )

        fig.update_layout(
            height=520,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            "Parametres: "
            f"bb_period={bb_period}, bb_std={bb_std:.2f}, entry_z={entry_z:.2f}, "
            f"atr_period={atr_period}, atr_percentile={atr_percentile:.0f}%, k_sl={k_sl:.2f}"
        )
        st.markdown(
            "- bb_period: fenetre de moyenne pour la ligne centrale.\n"
            "- bb_std: largeur des bandes (ecart-type).\n"
            "- entry_z: seuil d'entree en z-score (declenchement).\n"
            "- atr_period: fenetre ATR pour la volatilite.\n"
            "- atr_percentile: seuil de volatilite minimum.\n"
            "- k_sl: distance du stop = k_sl x ATR."
        )
        return

    if strategy_key == "bollinger_atr_v2":
        bb_period = max(2, min(int(params.get("bb_period", 20)), n - 1))
        bb_std = float(params.get("bb_std", 2.0))
        entry_z = float(params.get("entry_z", bb_std))
        atr_period = max(2, min(int(params.get("atr_period", 14)), n - 1))
        atr_percentile = float(params.get("atr_percentile", 30))
        bb_stop_factor = float(params.get("bb_stop_factor", 0.5))

        middle = price_series.rolling(window=bb_period, min_periods=1).mean()
        sigma = price_series.rolling(window=bb_period, min_periods=1).std(ddof=0).fillna(0.5)
        upper = middle + sigma * bb_std
        lower = middle - sigma * bb_std
        entry_upper = middle + sigma * entry_z
        entry_lower = middle - sigma * entry_z

        atr_values = price_series.diff().abs().rolling(window=atr_period, min_periods=1).mean()
        atr_threshold = float(np.nanpercentile(atr_values, atr_percentile))

        # Calculer les stop-loss Bollinger
        bb_distance_lower = middle - lower
        bb_distance_upper = upper - middle
        stop_long = lower - bb_stop_factor * bb_distance_lower
        stop_short = upper + bb_stop_factor * bb_distance_upper

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                "Prix + Bandes de Bollinger + Stop-Loss Bollinger (V2)",
                "ATR et filtre de volatilite",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                name="Bollinger bas",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                name="Bollinger haut",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
                fill="tonexty",
                fillcolor="rgba(100, 160, 200, 0.15)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=middle,
                name="Bollinger milieu",
                line=dict(color="rgba(140, 200, 255, 0.9)", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_upper,
                name="Seuil entry_z haut",
                line=dict(color="rgba(255, 204, 128, 0.9)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_lower,
                name="Seuil entry_z bas",
                line=dict(color="rgba(255, 204, 128, 0.9)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        # Ajouter les niveaux de stop-loss
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stop_long,
                name="Stop LONG",
                line=dict(color="rgba(239, 83, 80, 0.7)", width=1.2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stop_short,
                name="Stop SHORT",
                line=dict(color="rgba(239, 83, 80, 0.7)", width=1.2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            ),
            row=1,
            col=1,
        )

        entry_index = int(n * 0.25)
        entry_price = float(entry_lower.iloc[entry_index])
        stop_price = float(stop_long.iloc[entry_index])
        fig.add_trace(
            go.Scatter(
                x=[entry_index],
                y=[entry_price],
                mode="markers",
                name="Exemple entree LONG",
                marker=dict(color="#26a69a", size=8),
            ),
            row=1,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=entry_index,
            x1=entry_index,
            y0=entry_price,
            y1=stop_price,
            line=dict(color="#ef5350", width=2),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=entry_index,
            y=stop_price,
            text=f"Stop = lower - {bb_stop_factor:.1f} × (mid-low)",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=20,
            font=dict(color="#ef9a9a", size=10),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=atr_values,
                name="ATR",
                line=dict(color="#ab47bc", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=atr_threshold,
            line_dash="dot",
            line_color="#ffa726",
            annotation_text=f"Seuil {atr_percentile:.0f}%",
            annotation_position="top left",
            row=2,
            col=1,
        )

        fig.update_layout(
            height=520,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            "Parametres: "
            f"bb_period={bb_period}, bb_std={bb_std:.2f}, entry_z={entry_z:.2f}, "
            f"atr_period={atr_period}, atr_percentile={atr_percentile:.0f}%, bb_stop_factor={bb_stop_factor:.2f}"
        )
        st.markdown(
            "**V2: Stop-loss basé sur les Bandes de Bollinger**\n\n"
            "- bb_stop_factor: Distance du stop depuis les bandes (0.2=proche, 2.0=loin).\n"
            "- LONG: stop = lower - bb_stop_factor × (middle - lower)\n"
            "- SHORT: stop = upper + bb_stop_factor × (upper - middle)\n"
            "- Le stop-loss est FIXE au moment de l'entrée."
        )
        return

    if strategy_key == "bollinger_atr_v3":
        bb_period = max(2, min(int(params.get("bb_period", 20)), n - 1))
        bb_std = float(params.get("bb_std", 2.0))
        atr_period = max(2, min(int(params.get("atr_period", 14)), n - 1))
        atr_percentile = float(params.get("atr_percentile", 30))
        entry_pct_long = float(params.get("entry_pct_long", 0.0))
        entry_pct_short = float(params.get("entry_pct_short", 1.0))
        stop_factor = float(params.get("stop_factor", 0.5))
        tp_factor = float(params.get("tp_factor", 0.7))

        middle = price_series.rolling(window=bb_period, min_periods=1).mean()
        sigma = price_series.rolling(window=bb_period, min_periods=1).std(ddof=0).fillna(0.5)
        upper = middle + sigma * bb_std
        lower = middle - sigma * bb_std

        atr_values = price_series.diff().abs().rolling(window=atr_period, min_periods=1).mean()
        atr_threshold = float(np.nanpercentile(atr_values, atr_percentile))

        # Calculer les niveaux d'entrée sur l'échelle unifiée
        total_distance = upper - lower
        entry_level_long = lower + entry_pct_long * total_distance
        entry_level_short = lower + entry_pct_short * total_distance

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f"Prix + Bandes de Bollinger (V3: {bb_std:.1f}σ) + Niveaux d'entrée variables",
                "ATR et filtre de volatilite",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                name="Bollinger bas (0%)",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                name="Bollinger haut (100%)",
                line=dict(color="rgba(100, 160, 200, 0.6)", width=1),
                fill="tonexty",
                fillcolor="rgba(100, 160, 200, 0.15)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=middle,
                name="Bollinger milieu (50%)",
                line=dict(color="rgba(140, 200, 255, 0.9)", width=1.5),
            ),
            row=1,
            col=1,
        )
        # Niveaux d'entrée
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_level_long,
                name=f"Entrée LONG ({entry_pct_long*100:.0f}%)",
                line=dict(color="rgba(76, 175, 80, 0.9)", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=entry_level_short,
                name=f"Entrée SHORT ({entry_pct_short*100:.0f}%)",
                line=dict(color="rgba(171, 71, 188, 0.9)", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            ),
            row=1,
            col=1,
        )

        # Exemple de trade LONG
        entry_index = int(n * 0.25)
        entry_price = float(entry_level_long.iloc[entry_index])
        distance = float(total_distance.iloc[entry_index])
        stop_price = entry_price - stop_factor * distance
        tp_price = entry_price + tp_factor * distance

        fig.add_trace(
            go.Scatter(
                x=[entry_index],
                y=[entry_price],
                mode="markers",
                name="Exemple entree LONG",
                marker=dict(color="#26a69a", size=8),
            ),
            row=1,
            col=1,
        )
        # Ligne Stop
        fig.add_shape(
            type="line",
            x0=entry_index,
            x1=entry_index + 15,
            y0=stop_price,
            y1=stop_price,
            line=dict(color="#ef5350", width=2, dash="dash"),
            row=1,
            col=1,
        )
        # Ligne TP
        fig.add_shape(
            type="line",
            x0=entry_index,
            x1=entry_index + 15,
            y0=tp_price,
            y1=tp_price,
            line=dict(color="#4caf50", width=2, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=entry_index + 15,
            y=stop_price,
            text=f"Stop ({stop_factor*100:.0f}%)",
            showarrow=False,
            xshift=45,
            font=dict(color="#ef9a9a", size=9),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=entry_index + 15,
            y=tp_price,
            text=f"TP ({tp_factor*100:.0f}%)",
            showarrow=False,
            xshift=35,
            font=dict(color="#81c784", size=9),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=atr_values,
                name="ATR",
                line=dict(color="#ab47bc", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=atr_threshold,
            line_dash="dot",
            line_color="#ffa726",
            annotation_text=f"Seuil {atr_percentile:.0f}%",
            annotation_position="top left",
            row=2,
            col=1,
        )

        fig.update_layout(
            height=520,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            "Parametres: "
            f"bb_period={bb_period}, bb_std={bb_std:.2f}, "
            f"entry_long={entry_pct_long*100:.0f}%, entry_short={entry_pct_short*100:.0f}%, "
            f"stop_factor={stop_factor:.2f}, tp_factor={tp_factor:.2f}"
        )
        st.markdown(
            "**V3: Entrées, Stop et TP Variables sur Échelle Unifiée**\n\n"
            "- **Échelle**: 0% = lower_band, 50% = middle_band, 100% = upper_band\n"
            "- **entry_pct_long**: Position d'entrée LONG (-50% à +20%)\n"
            "- **entry_pct_short**: Position d'entrée SHORT (+80% à +150%)\n"
            "- **stop_factor**: Distance stop depuis entry_price (10% à 100% de distance totale)\n"
            "- **tp_factor**: Distance TP depuis entry_price (20% à 150% de distance totale)\n"
            "- **bb_std**: Amplitude des bandes (1σ à 4σ)\n\n"
            "Formule: entry_level = lower + entry_pct × (upper - lower)"
        )
        return

    if strategy_key == "ema_cross":
        fast_period = max(2, min(int(params.get("fast_period", 12)), n - 1))
        slow_period = max(3, min(int(params.get("slow_period", 26)), n - 1))
        ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
        diff = ema_fast - ema_slow
        cross_idx = diff.diff().fillna(0)
        cross_points = cross_idx[cross_idx != 0].index.tolist()
        marker_index = int(cross_points[0]) if cross_points else int(n * 0.55)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ema_fast,
                name=f"EMA rapide ({fast_period})",
                line=dict(color="#42a5f5", width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ema_slow,
                name=f"EMA lente ({slow_period})",
                line=dict(color="#ffb74d", width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[marker_index],
                y=[price[marker_index]],
                mode="markers",
                name="Exemple croisement",
                marker=dict(color="#26a69a", size=8),
            )
        )

        fig.update_layout(
            height=460,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            title="EMA Cross: croisement de moyennes mobiles",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(f"Parametres: fast_period={fast_period}, slow_period={slow_period}")
        st.markdown(
            "- fast_period: vitesse de la moyenne rapide.\n"
            "- slow_period: tendance de fond (plus lente).\n"
            "- Un signal apparait quand la rapide croise la lente."
        )
        return

    if strategy_key == "macd_cross":
        fast_period = max(2, min(int(params.get("fast_period", 12)), n - 1))
        slow_period = max(3, min(int(params.get("slow_period", 26)), n - 1))
        signal_period = max(2, min(int(params.get("signal_period", 9)), n - 1))
        ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("Prix", "MACD (ligne vs signal)"),
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=macd_line,
                name="MACD",
                line=dict(color="#26a69a", width=1.6),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=signal_line,
                name="Signal",
                line=dict(color="#ef5350", width=1.4),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=520,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            f"Parametres: fast_period={fast_period}, slow_period={slow_period}, "
            f"signal_period={signal_period}"
        )
        st.markdown(
            "- fast_period: EMA rapide pour le MACD.\n"
            "- slow_period: EMA lente pour le MACD.\n"
            "- signal_period: lissage de la ligne MACD.\n"
            "- Signal quand MACD croise la ligne signal."
        )
        return

    if strategy_key == "rsi_reversal":
        rsi_period = max(2, min(int(params.get("rsi_period", 14)), n - 1))
        oversold = float(params.get("oversold_level", 30))
        overbought = float(params.get("overbought_level", 70))

        delta = price_series.diff().fillna(0)
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / rsi_period, adjust=False).mean().replace(0, 1e-6)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Prix", "RSI et zones extremes"),
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rsi,
                name="RSI",
                line=dict(color="#42a5f5", width=1.6),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=oversold,
            line_dash="dot",
            line_color="#26a69a",
            annotation_text="Oversold",
            annotation_position="bottom left",
            row=2,
            col=1,
        )
        fig.add_hline(
            y=overbought,
            line_dash="dot",
            line_color="#ef5350",
            annotation_text="Overbought",
            annotation_position="top left",
            row=2,
            col=1,
        )

        fig.update_layout(
            height=520,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            f"Parametres: rsi_period={rsi_period}, oversold={oversold:.0f}, overbought={overbought:.0f}"
        )
        st.markdown(
            "- rsi_period: fenetre de calcul du RSI.\n"
            "- oversold_level: seuil bas pour signal long.\n"
            "- overbought_level: seuil haut pour signal short."
        )
        return

    if strategy_key == "atr_channel":
        atr_period = max(2, min(int(params.get("atr_period", 14)), n - 1))
        atr_mult = float(params.get("atr_mult", 2.0))
        ema_center = price_series.ewm(span=atr_period, adjust=False).mean()
        high = price_series + 0.6 + 0.3 * np.sin(np.linspace(0, 8 * np.pi, n))
        low = price_series - 0.6 - 0.3 * np.sin(np.linspace(0, 8 * np.pi, n))
        prev_close = price_series.shift(1).fillna(price_series.iloc[0])
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_values = tr.ewm(span=atr_period, adjust=False).mean()
        upper = ema_center + atr_values * atr_mult
        lower = ema_center - atr_values * atr_mult

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=price,
                name="Prix",
                line=dict(color="#e0e0e0", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                name="Canal haut",
                line=dict(color="#ef5350", width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                name="Canal bas",
                line=dict(color="#26a69a", width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ema_center,
                name="EMA centre",
                line=dict(color="#42a5f5", width=1.4, dash="dot"),
            )
        )

        fig.update_layout(
            height=460,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8b2d1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            title="ATR Channel: EMA +/- ATR * multiplier",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        _apply_axis_interaction(fig)
        st.plotly_chart(
            fig, width="stretch", key=key, config=PLOTLY_CHART_CONFIG
        )
        st.caption(
            f"Parametres: atr_period={atr_period}, atr_mult={atr_mult:.2f}"
        )
        st.markdown(
            "- atr_period: fenetre ATR et EMA du canal.\n"
            "- atr_mult: largeur du canal (volatilite)."
        )
        return

    st.info("Schema non disponible pour cette strategie.")


def render_trade_pnl_distribution(
    trades_df: pd.DataFrame,
    title: str = "📊 Distribution des P&L par Trade",
    key: str = "trade_pnl_dist",
    height: int = 400,
) -> None:
    """
    Affiche la distribution des P&L par trade avec histogramme + KDE (seaborn).

    Args:
        trades_df: DataFrame des trades avec colonne 'pnl'
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if not SEABORN_AVAILABLE:
        st.warning("⚠️ Seaborn non disponible - Distribution non affichée")
        return

    if trades_df.empty or 'pnl' not in trades_df.columns:
        st.warning("⚠️ Aucune donnée de P&L à afficher")
        return

    st.markdown(f"#### {title}")
    st.caption("📊 Graphique généré avec **Seaborn** (histogramme + KDE)")

    # Configuration style seaborn
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, height / 100))

    # Histogramme + KDE avec seaborn
    pnl_values = trades_df['pnl'].dropna()
    sns.histplot(
        pnl_values,
        kde=True,
        color="#26a69a",
        edgecolor="#1e272e",
        alpha=0.7,
        ax=ax,
        stat="density"
    )

    # Ligne verticale à zéro
    ax.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Statistiques
    mean_pnl = pnl_values.mean()
    median_pnl = pnl_values.median()
    ax.axvline(mean_pnl, color='#ffa726', linestyle='-', linewidth=2, label=f'Moyenne: ${mean_pnl:.2f}')
    ax.axvline(median_pnl, color='#42a5f5', linestyle='-', linewidth=2, label=f'Médiane: ${median_pnl:.2f}')

    ax.set_xlabel('P&L ($)', color='#a8b2d1', fontsize=11)
    ax.set_ylabel('Densité', color='#a8b2d1', fontsize=11)
    ax.set_title(title, color='#a8b2d1', fontsize=13, pad=15)
    ax.legend(loc='upper right', facecolor='#1e272e', edgecolor='#a8b2d1', labelcolor='#a8b2d1')

    # Style sombre
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e272e')
    ax.tick_params(colors='#a8b2d1')
    ax.spines['bottom'].set_color('#a8b2d1')
    ax.spines['top'].set_color('#a8b2d1')
    ax.spines['right'].set_color('#a8b2d1')
    ax.spines['left'].set_color('#a8b2d1')

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_returns_distribution(
    returns: pd.Series,
    title: str = "📈 Distribution des Rendements",
    key: str = "returns_dist",
    height: int = 400,
) -> None:
    """
    Affiche la distribution des rendements avec histogramme + KDE (seaborn).

    Args:
        returns: Série des rendements
        title: Titre du graphique
        key: Clé unique Streamlit
        height: Hauteur du graphique
    """
    if not SEABORN_AVAILABLE:
        st.warning("⚠️ Seaborn non disponible - Distribution non affichée")
        return

    if returns.empty:
        st.warning("⚠️ Aucune donnée de rendements à afficher")
        return

    st.markdown(f"#### {title}")
    st.caption("📊 Graphique généré avec **Seaborn** (histogramme + KDE)")

    # Configuration style seaborn
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, height / 100))

    # Histogramme + KDE
    returns_clean = returns.dropna()
    sns.histplot(
        returns_clean,
        kde=True,
        color="#42a5f5",
        edgecolor="#1e272e",
        alpha=0.7,
        ax=ax,
        stat="density"
    )

    # Ligne verticale à zéro
    ax.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Statistiques
    mean_ret = returns_clean.mean()
    std_ret = returns_clean.std()
    ax.axvline(mean_ret, color='#26a69a', linestyle='-', linewidth=2, label=f'Moyenne: {mean_ret:.4f}')
    ax.axvline(mean_ret + std_ret, color='#ef5350', linestyle=':', linewidth=1.5, label=f'+1σ: {mean_ret + std_ret:.4f}')
    ax.axvline(mean_ret - std_ret, color='#ef5350', linestyle=':', linewidth=1.5, label=f'-1σ: {mean_ret - std_ret:.4f}')

    ax.set_xlabel('Rendement', color='#a8b2d1', fontsize=11)
    ax.set_ylabel('Densité', color='#a8b2d1', fontsize=11)
    ax.set_title(title, color='#a8b2d1', fontsize=13, pad=15)
    ax.legend(loc='upper right', facecolor='#1e272e', edgecolor='#a8b2d1', labelcolor='#a8b2d1')

    # Style sombre
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e272e')
    ax.tick_params(colors='#a8b2d1')
    ax.spines['bottom'].set_color('#a8b2d1')
    ax.spines['top'].set_color('#a8b2d1')
    ax.spines['right'].set_color('#a8b2d1')
    ax.spines['left'].set_color('#a8b2d1')

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
    "render_ohlcv_with_trades_and_indicators",
    "render_ohlcv_with_indicators",
    "render_equity_curve",
    "render_comparison_chart",
    "render_strategy_param_diagram",
    "render_trade_pnl_distribution",
    "render_returns_distribution",
]
