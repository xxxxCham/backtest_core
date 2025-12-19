"""
Indicator Explorer - Visualisation interactive des indicateurs.

Composant Phase 5.3 pour explorer visuellement les indicateurs techniques
avec overlay sur les prix OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class IndicatorType(Enum):
    """Type d'indicateur pour le placement sur le graphique."""
    OVERLAY = "overlay"       # Sur le prix (MA, Bollinger, etc.)
    OSCILLATOR = "oscillator" # Panel séparé (RSI, MACD, etc.)
    VOLUME = "volume"         # Panel volume


@dataclass
class IndicatorConfig:
    """Configuration d'un indicateur à afficher."""
    name: str
    indicator_fn: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    indicator_type: IndicatorType = IndicatorType.OVERLAY
    color: str = "#1f77b4"
    line_width: int = 1
    opacity: float = 1.0
    show_by_default: bool = True
    
    # Pour les indicateurs multi-lignes (Bollinger, MACD)
    secondary_colors: List[str] = field(default_factory=lambda: ["#ff7f0e", "#2ca02c", "#d62728"])


@dataclass
class ChartConfig:
    """Configuration globale du graphique."""
    height: int = 800
    show_volume: bool = True
    candlestick_colors: Tuple[str, str] = ("#26a69a", "#ef5350")  # Up, Down
    background_color: str = "#0e1117"
    grid_color: str = "#1e2130"
    text_color: str = "#fafafa"
    range_slider: bool = False
    
    
# Configurations par défaut pour les indicateurs connus
DEFAULT_INDICATOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sma": {"type": IndicatorType.OVERLAY, "color": "#ff9800"},
    "ema": {"type": IndicatorType.OVERLAY, "color": "#2196f3"},
    "bollinger": {"type": IndicatorType.OVERLAY, "color": "#9c27b0"},
    "keltner": {"type": IndicatorType.OVERLAY, "color": "#00bcd4"},
    "donchian": {"type": IndicatorType.OVERLAY, "color": "#4caf50"},
    "supertrend": {"type": IndicatorType.OVERLAY, "color": "#e91e63"},
    "ichimoku": {"type": IndicatorType.OVERLAY, "color": "#ff5722"},
    "psar": {"type": IndicatorType.OVERLAY, "color": "#ffeb3b"},
    "rsi": {"type": IndicatorType.OSCILLATOR, "color": "#9c27b0", "levels": [30, 70]},
    "stochastic": {"type": IndicatorType.OSCILLATOR, "color": "#2196f3", "levels": [20, 80]},
    "stoch_rsi": {"type": IndicatorType.OSCILLATOR, "color": "#00bcd4", "levels": [20, 80]},
    "williams_r": {"type": IndicatorType.OSCILLATOR, "color": "#ff9800", "levels": [-80, -20]},
    "cci": {"type": IndicatorType.OSCILLATOR, "color": "#4caf50", "levels": [-100, 100]},
    "mfi": {"type": IndicatorType.OSCILLATOR, "color": "#e91e63", "levels": [20, 80]},
    "macd": {"type": IndicatorType.OSCILLATOR, "color": "#2196f3"},
    "adx": {"type": IndicatorType.OSCILLATOR, "color": "#ff5722", "levels": [25]},
    "momentum": {"type": IndicatorType.OSCILLATOR, "color": "#9c27b0"},
    "roc": {"type": IndicatorType.OSCILLATOR, "color": "#00bcd4"},
    "vortex": {"type": IndicatorType.OSCILLATOR, "color": "#4caf50"},
    "aroon": {"type": IndicatorType.OSCILLATOR, "color": "#ff9800"},
    "atr": {"type": IndicatorType.OSCILLATOR, "color": "#e91e63"},
    "obv": {"type": IndicatorType.VOLUME, "color": "#2196f3"},
    "vwap": {"type": IndicatorType.OVERLAY, "color": "#ffeb3b"},
}


class IndicatorExplorer:
    """
    Explorateur interactif d'indicateurs techniques.
    
    Permet de visualiser les indicateurs sur un graphique OHLCV
    avec configuration dynamique des paramètres.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        chart_config: Optional[ChartConfig] = None,
    ):
        """
        Initialise l'explorateur.
        
        Args:
            df: DataFrame OHLCV avec colonnes open, high, low, close, volume
            chart_config: Configuration du graphique
        """
        self.df = df.copy()
        self.config = chart_config or ChartConfig()
        self._indicators: Dict[str, Dict[str, Any]] = {}
        self._computed_values: Dict[str, Any] = {}
        
        # Vérifier colonnes requises
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
    
    def add_indicator(
        self,
        name: str,
        values: Union[np.ndarray, pd.Series, Dict[str, np.ndarray]],
        indicator_type: Optional[IndicatorType] = None,
        color: str = "#1f77b4",
        label: Optional[str] = None,
        **kwargs,
    ) -> "IndicatorExplorer":
        """
        Ajoute un indicateur calculé.
        
        Args:
            name: Nom unique de l'indicateur
            values: Valeurs (array, Series, ou dict pour multi-lignes)
            indicator_type: Type (overlay, oscillator, volume)
            color: Couleur principale
            label: Label d'affichage
            **kwargs: Options supplémentaires
            
        Returns:
            Self pour chaînage
        """
        # Déterminer le type automatiquement si non spécifié
        if indicator_type is None:
            default_config = DEFAULT_INDICATOR_CONFIGS.get(name.lower(), {})
            indicator_type = default_config.get("type", IndicatorType.OVERLAY)
        
        self._indicators[name] = {
            "values": values,
            "type": indicator_type,
            "color": color,
            "label": label or name,
            **kwargs,
        }
        
        return self
    
    def remove_indicator(self, name: str) -> "IndicatorExplorer":
        """Supprime un indicateur."""
        self._indicators.pop(name, None)
        return self
    
    def clear_indicators(self) -> "IndicatorExplorer":
        """Supprime tous les indicateurs."""
        self._indicators.clear()
        return self
    
    def _create_candlestick(self, fig: go.Figure, row: int = 1) -> None:
        """Ajoute le candlestick au graphique."""
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df["open"],
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                increasing_line_color=self.config.candlestick_colors[0],
                decreasing_line_color=self.config.candlestick_colors[1],
                name="OHLC",
            ),
            row=row,
            col=1,
        )
    
    def _add_overlay_indicator(
        self,
        fig: go.Figure,
        name: str,
        config: Dict[str, Any],
        row: int = 1,
    ) -> None:
        """Ajoute un indicateur overlay sur le prix."""
        values = config["values"]
        color = config["color"]
        label = config["label"]
        
        if isinstance(values, dict):
            # Multi-lignes (ex: Bollinger bands)
            colors = [color] + config.get("secondary_colors", ["#ff7f0e", "#2ca02c", "#d62728"])
            for i, (key, val) in enumerate(values.items()):
                fig.add_trace(
                    go.Scatter(
                        x=self.df.index,
                        y=val,
                        mode="lines",
                        name=f"{label} {key}",
                        line=dict(color=colors[i % len(colors)], width=1),
                        opacity=config.get("opacity", 0.8),
                    ),
                    row=row,
                    col=1,
                )
        else:
            # Ligne simple
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=values,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=config.get("line_width", 1)),
                    opacity=config.get("opacity", 1.0),
                ),
                row=row,
                col=1,
            )
    
    def _add_oscillator_indicator(
        self,
        fig: go.Figure,
        name: str,
        config: Dict[str, Any],
        row: int,
    ) -> None:
        """Ajoute un indicateur oscillateur dans un panel séparé."""
        values = config["values"]
        color = config["color"]
        label = config["label"]
        levels = config.get("levels", [])
        
        if isinstance(values, dict):
            # Multi-lignes (ex: MACD)
            colors = [color, "#ff7f0e", "#2ca02c"]
            for i, (key, val) in enumerate(values.items()):
                # Histogram spécial pour MACD
                if key == "histogram" and "macd" in name.lower():
                    colors_hist = np.where(val >= 0, "#26a69a", "#ef5350")
                    fig.add_trace(
                        go.Bar(
                            x=self.df.index,
                            y=val,
                            name=f"{label} {key}",
                            marker_color=colors_hist.tolist() if hasattr(colors_hist, 'tolist') else list(colors_hist),
                        ),
                        row=row,
                        col=1,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=self.df.index,
                            y=val,
                            mode="lines",
                            name=f"{label} {key}",
                            line=dict(color=colors[i % len(colors)], width=1),
                        ),
                        row=row,
                        col=1,
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=values,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1),
                ),
                row=row,
                col=1,
            )
        
        # Ajouter les niveaux de référence
        for level in levels:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=row,
                col=1,
            )
    
    def _add_volume(self, fig: go.Figure, row: int) -> None:
        """Ajoute le volume."""
        if "volume" not in self.df.columns:
            return
            
        colors = np.where(
            self.df["close"] >= self.df["open"],
            self.config.candlestick_colors[0],
            self.config.candlestick_colors[1],
        )
        
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df["volume"],
                name="Volume",
                marker_color=colors.tolist(),
                opacity=0.7,
            ),
            row=row,
            col=1,
        )
    
    def create_figure(self) -> go.Figure:
        """
        Crée la figure Plotly complète.
        
        Returns:
            Figure Plotly avec tous les indicateurs
        """
        # Compter les panels nécessaires
        oscillators = [
            (name, cfg) for name, cfg in self._indicators.items()
            if cfg["type"] == IndicatorType.OSCILLATOR
        ]
        
        n_rows = 1  # Prix
        if self.config.show_volume and "volume" in self.df.columns:
            n_rows += 1
        n_rows += len(oscillators)
        
        # Créer les row heights
        row_heights = [0.5]  # Prix
        if self.config.show_volume and "volume" in self.df.columns:
            row_heights.append(0.1)
        row_heights.extend([0.15] * len(oscillators))
        
        # Normaliser
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]
        
        # Créer la figure avec subplots
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
        )
        
        # Ajouter le candlestick
        self._create_candlestick(fig, row=1)
        
        # Ajouter les overlays
        for name, config in self._indicators.items():
            if config["type"] == IndicatorType.OVERLAY:
                self._add_overlay_indicator(fig, name, config, row=1)
        
        current_row = 2
        
        # Ajouter le volume
        if self.config.show_volume and "volume" in self.df.columns:
            self._add_volume(fig, row=current_row)
            current_row += 1
        
        # Ajouter les oscillateurs
        for name, config in oscillators:
            self._add_oscillator_indicator(fig, name, config, row=current_row)
            current_row += 1
        
        # Styling
        fig.update_layout(
            height=self.config.height,
            template="plotly_dark",
            paper_bgcolor=self.config.background_color,
            plot_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            xaxis_rangeslider_visible=self.config.range_slider,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=50, r=50, t=50, b=50),
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.config.grid_color,
            showgrid=True,
        )
        fig.update_yaxes(
            gridcolor=self.config.grid_color,
            showgrid=True,
        )
        
        return fig
    
    def get_indicator_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne un résumé des indicateurs ajoutés.
        
        Returns:
            Dict avec infos sur chaque indicateur
        """
        summary = {}
        for name, config in self._indicators.items():
            values = config["values"]
            
            if isinstance(values, dict):
                stats = {}
                for key, val in values.items():
                    arr = np.asarray(val)
                    valid = arr[~np.isnan(arr)]
                    if len(valid) > 0:
                        stats[key] = {
                            "current": float(valid[-1]),
                            "min": float(np.min(valid)),
                            "max": float(np.max(valid)),
                            "mean": float(np.mean(valid)),
                        }
            else:
                arr = np.asarray(values)
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    stats = {
                        "current": float(valid[-1]),
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                    }
                else:
                    stats = {}
            
            summary[name] = {
                "type": config["type"].value,
                "label": config["label"],
                "stats": stats,
            }
        
        return summary


def render_indicator_explorer(
    df: pd.DataFrame,
    available_indicators: Optional[Dict[str, Callable]] = None,
    key: str = "indicator_explorer",
) -> Optional[go.Figure]:
    """
    Rendu Streamlit de l'explorateur d'indicateurs.
    
    Args:
        df: DataFrame OHLCV
        available_indicators: Dict {nom: fonction} des indicateurs disponibles
        key: Clé unique pour les widgets Streamlit
        
    Returns:
        Figure Plotly ou None
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
    # Import du registre d'indicateurs si disponible
    try:
        from indicators.registry import INDICATOR_REGISTRY
        if available_indicators is None:
            available_indicators = dict(INDICATOR_REGISTRY)
    except ImportError:
        if available_indicators is None:
            available_indicators = {}
    
    st.subheader("📊 Explorateur d'Indicateurs")
    
    # Configuration du graphique
    with st.expander("⚙️ Configuration graphique", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            height = st.slider("Hauteur", 400, 1200, 800, key=f"{key}_height")
        with col2:
            show_volume = st.checkbox("Afficher volume", value=True, key=f"{key}_volume")
        with col3:
            range_slider = st.checkbox("Range slider", value=False, key=f"{key}_slider")
    
    chart_config = ChartConfig(
        height=height,
        show_volume=show_volume,
        range_slider=range_slider,
    )
    
    explorer = IndicatorExplorer(df, chart_config)
    
    # Sélection des indicateurs
    st.markdown("### 📈 Sélection des indicateurs")
    
    # Grouper par type
    overlay_indicators = []
    oscillator_indicators = []
    
    for name in available_indicators.keys():
        default_config = DEFAULT_INDICATOR_CONFIGS.get(name.lower(), {})
        ind_type = default_config.get("type", IndicatorType.OVERLAY)
        if ind_type == IndicatorType.OSCILLATOR:
            oscillator_indicators.append(name)
        else:
            overlay_indicators.append(name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Overlays (sur le prix)**")
        selected_overlays = st.multiselect(
            "Choisir overlays",
            overlay_indicators,
            default=[],
            key=f"{key}_overlays",
            label_visibility="collapsed",
        )
    
    with col2:
        st.markdown("**Oscillateurs (panels séparés)**")
        selected_oscillators = st.multiselect(
            "Choisir oscillateurs",
            oscillator_indicators,
            default=[],
            key=f"{key}_oscillators",
            label_visibility="collapsed",
        )
    
    # Configuration des paramètres pour chaque indicateur sélectionné
    all_selected = selected_overlays + selected_oscillators
    
    if all_selected:
        st.markdown("### 🔧 Paramètres des indicateurs")
        
        for ind_name in all_selected:
            with st.expander(f"📊 {ind_name}", expanded=False):
                indicator_fn = available_indicators.get(ind_name)
                if indicator_fn is None:
                    st.warning(f"Indicateur {ind_name} non disponible")
                    continue
                
                # Paramètres par défaut selon l'indicateur
                params = {}
                
                if "period" in ind_name.lower() or ind_name.lower() in ["sma", "ema", "rsi", "atr"]:
                    params["period"] = st.slider(
                        f"Période",
                        5, 200, 14,
                        key=f"{key}_{ind_name}_period",
                    )
                elif ind_name.lower() == "bollinger":
                    params["window"] = st.slider("Window", 5, 50, 20, key=f"{key}_{ind_name}_window")
                    params["num_std"] = st.slider("Std Dev", 1.0, 4.0, 2.0, 0.1, key=f"{key}_{ind_name}_std")
                elif ind_name.lower() == "macd":
                    params["fast_period"] = st.slider("Fast", 5, 20, 12, key=f"{key}_{ind_name}_fast")
                    params["slow_period"] = st.slider("Slow", 15, 50, 26, key=f"{key}_{ind_name}_slow")
                    params["signal_period"] = st.slider("Signal", 5, 20, 9, key=f"{key}_{ind_name}_signal")
                elif ind_name.lower() == "stochastic":
                    params["k_period"] = st.slider("K Period", 5, 30, 14, key=f"{key}_{ind_name}_k")
                    params["d_period"] = st.slider("D Period", 1, 10, 3, key=f"{key}_{ind_name}_d")
                
                # Calculer l'indicateur
                try:
                    # Préparer les données
                    if "volume" in df.columns:
                        result = indicator_fn(
                            df["high"].values,
                            df["low"].values,
                            df["close"].values,
                            **params
                        ) if ind_name.lower() not in ["sma", "ema", "rsi", "momentum", "roc"] else indicator_fn(
                            df["close"].values,
                            **params
                        )
                    else:
                        result = indicator_fn(df["close"].values, **params)
                    
                    # Déterminer le type
                    default_config = DEFAULT_INDICATOR_CONFIGS.get(ind_name.lower(), {})
                    ind_type = default_config.get("type", IndicatorType.OVERLAY)
                    color = default_config.get("color", "#1f77b4")
                    
                    explorer.add_indicator(
                        ind_name,
                        result,
                        indicator_type=ind_type,
                        color=color,
                        levels=default_config.get("levels", []),
                    )
                    
                    st.success(f"✅ {ind_name} calculé")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    # Créer et afficher le graphique
    if st.button("🚀 Générer graphique", key=f"{key}_generate", type="primary"):
        with st.spinner("Génération du graphique..."):
            fig = explorer.create_figure()
            st.plotly_chart(fig, width='stretch', key=f"{key}_chart")
            
            # Résumé des indicateurs
            summary = explorer.get_indicator_summary()
            if summary:
                st.markdown("### 📋 Résumé des indicateurs")
                for name, info in summary.items():
                    with st.expander(f"📊 {info['label']}", expanded=False):
                        if isinstance(info["stats"], dict) and "current" in info["stats"]:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Actuel", f"{info['stats']['current']:.4f}")
                            col2.metric("Min", f"{info['stats']['min']:.4f}")
                            col3.metric("Max", f"{info['stats']['max']:.4f}")
                            col4.metric("Moyenne", f"{info['stats']['mean']:.4f}")
                        elif isinstance(info["stats"], dict):
                            for key, stats in info["stats"].items():
                                if isinstance(stats, dict):
                                    st.write(f"**{key}**: {stats.get('current', 'N/A'):.4f}")
            
            return fig
    
    return None


def render_quick_indicator_chart(
    df: pd.DataFrame,
    indicators: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    title: str = "Indicateurs",
    height: int = 600,
) -> go.Figure:
    """
    Génère rapidement un graphique avec indicateurs pré-calculés.
    
    Args:
        df: DataFrame OHLCV
        indicators: Dict {nom: valeurs} des indicateurs
        title: Titre du graphique
        height: Hauteur en pixels
        
    Returns:
        Figure Plotly
    """
    config = ChartConfig(height=height, show_volume=True)
    explorer = IndicatorExplorer(df, config)
    
    for name, values in indicators.items():
        default_config = DEFAULT_INDICATOR_CONFIGS.get(name.lower(), {})
        ind_type = default_config.get("type", IndicatorType.OVERLAY)
        color = default_config.get("color", "#1f77b4")
        
        explorer.add_indicator(
            name,
            values,
            indicator_type=ind_type,
            color=color,
            levels=default_config.get("levels", []),
        )
    
    fig = explorer.create_figure()
    fig.update_layout(title=title)
    
    return fig
