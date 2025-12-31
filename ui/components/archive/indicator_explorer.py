"""
Module-ID: ui.components.archive.indicator_explorer

Purpose: Explorateur indicateurs interactif Phase 5.3 - visualiser indicateurs techniques avec overlay prix OHLCV.

Role in pipeline: visualization (archive)

Key components: IndicatorExplorer, render_explorer(), dynamique param inputs

Inputs: DataFrame OHLCV, indicateur sélectionné, params

Outputs: Graphique Plotly interactif (candlesticks + indicateur)

Dependencies: streamlit, plotly, indicators.registry

Conventions: Composant Streamlit optionnel

Read-if: Vous voulez explorer/visualiser indicateurs.

Skip-if: Archive - utiliser composants actifs ui/components/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from indicators.registry import calculate_indicator, get_indicator, list_indicators
    REGISTRY_AVAILABLE = True
except Exception:
    REGISTRY_AVAILABLE = False

from utils.indicator_ranges import get_indicator_param_specs, load_indicator_ranges


class IndicatorType(Enum):
    """Type d'indicateur pour le placement sur le graphique."""
    OVERLAY = "overlay"        # Sur le prix (MA, Bollinger, etc.)
    OSCILLATOR = "oscillator"  # Panel séparé (RSI, MACD, etc.)
    VOLUME = "volume"          # Panel volume


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
    "volume_oscillator": {"type": IndicatorType.VOLUME, "color": "#8bc34a"},
    "standard_deviation": {"type": IndicatorType.OSCILLATOR, "color": "#795548"},
    "fibonacci_levels": {"type": IndicatorType.OVERLAY, "color": "#ff9800"},
    "pivot_points": {"type": IndicatorType.OVERLAY, "color": "#9e9e9e"},
    "onchain_smoothing": {"type": IndicatorType.OVERLAY, "color": "#03a9f4"},
    "fear_greed": {"type": IndicatorType.OSCILLATOR, "color": "#ffc107", "levels": [20, 80]},
    "pi_cycle": {"type": IndicatorType.OVERLAY, "color": "#ff5722"},
    "amplitude_hunter": {"type": IndicatorType.OSCILLATOR, "color": "#00bcd4"},
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


def _clamp_value(value: float, min_val: float, max_val: float) -> float:
    if value is None:
        return min_val
    return max(min_val, min(max_val, value))


def _is_int_spec(spec: Dict[str, Any]) -> bool:
    min_val = spec.get("min")
    max_val = spec.get("max")
    default = spec.get("default")
    step = spec.get("step")
    values = [min_val, max_val, default]
    if any(val is None for val in values):
        return False
    if not all(isinstance(val, int) for val in values):
        return False
    return step is None or isinstance(step, int)


def _render_param_widget_from_spec(
    indicator_name: str,
    param_name: str,
    spec: Dict[str, Any],
    key_prefix: str,
) -> Any:
    label = spec.get("label", param_name)
    description = spec.get("description", "")
    default = spec.get("default")
    options = spec.get("options")
    param_type = spec.get("type")
    widget_key = f"{key_prefix}_{indicator_name}_{param_name}"

    if options:
        options_list = list(options)
        if default not in options_list and options_list:
            default = options_list[0]
        index = options_list.index(default) if default in options_list else 0
        return st.selectbox(
            label,
            options_list,
            index=index,
            key=widget_key,
            help=description,
        )

    if param_type == "string":
        return st.text_input(
            label,
            value="" if default is None else str(default),
            key=widget_key,
            help=description,
        )

    if param_type == "bool" or isinstance(default, bool):
        return st.checkbox(
            label,
            value=bool(default),
            key=widget_key,
            help=description,
        )

    min_val = spec.get("min")
    max_val = spec.get("max")
    step = spec.get("step")

    if min_val is None or max_val is None:
        if isinstance(default, int):
            return st.number_input(
                label,
                value=int(default),
                step=1,
                key=widget_key,
                help=description,
            )
        step_value = float(step) if step is not None else 0.1
        return st.number_input(
            label,
            value=float(default) if default is not None else 0.0,
            step=step_value,
            key=widget_key,
            help=description,
        )

    if _is_int_spec(spec):
        min_val = int(min_val)
        max_val = int(max_val)
        default_value = int(_clamp_value(default, min_val, max_val))
        step_value = int(step) if step is not None else 1
        return st.slider(
            label,
            min_val,
            max_val,
            value=default_value,
            step=step_value,
            key=widget_key,
            help=description,
        )

    min_val = float(min_val)
    max_val = float(max_val)
    default_value = float(_clamp_value(default, min_val, max_val))
    if step is not None:
        return st.slider(
            label,
            min_val,
            max_val,
            value=default_value,
            step=float(step),
            key=widget_key,
            help=description,
        )
    return st.slider(
        label,
        min_val,
        max_val,
        value=default_value,
        key=widget_key,
        help=description,
    )


def _build_params_from_specs(
    indicator_name: str,
    key_prefix: str,
    param_specs: Dict[str, Dict[str, Any]],
    allowed_params: Optional[set[str]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for param_name, spec in param_specs.items():
        if allowed_params is not None and param_name not in allowed_params:
            continue
        params[param_name] = _render_param_widget_from_spec(
            indicator_name,
            param_name,
            spec,
            key_prefix,
        )
    return params


def _build_params_from_settings(
    indicator_name: str,
    key_prefix: str,
    settings_class: Optional[type],
    exclude: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not settings_class or not is_dataclass(settings_class):
        return {}

    exclude = exclude or set()
    params: Dict[str, Any] = {}
    for param in fields(settings_class):
        if param.name in exclude:
            continue
        if param.default is not MISSING:
            default = param.default
        elif param.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            default = param.default_factory()  # type: ignore[misc]
        else:
            continue

        widget_key = f"{key_prefix}_{indicator_name}_{param.name}"
        label = param.name

        if isinstance(default, bool):
            params[param.name] = st.checkbox(
                label,
                value=default,
                key=widget_key,
            )
        elif isinstance(default, int):
            params[param.name] = st.number_input(
                label,
                value=default,
                step=1,
                key=widget_key,
            )
        elif isinstance(default, float):
            params[param.name] = st.number_input(
                label,
                value=default,
                step=0.1,
                key=widget_key,
            )
        elif isinstance(default, str):
            params[param.name] = st.text_input(
                label,
                value=default,
                key=widget_key,
            )

    return params


def _normalize_indicator_result(indicator_name: str, result: Any) -> Any:
    if isinstance(result, tuple):
        if indicator_name.lower() == "bollinger" and len(result) == 3:
            return {"upper": result[0], "middle": result[1], "lower": result[2]}
        if indicator_name.lower() == "stochastic" and len(result) == 2:
            return {"k": result[0], "d": result[1]}
        return {f"line_{idx + 1}": value for idx, value in enumerate(result)}
    return result


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

    indicator_ranges = load_indicator_ranges()
    use_registry = REGISTRY_AVAILABLE and available_indicators is None
    indicator_infos: Dict[str, Any] = {}

    if use_registry:
        indicator_names = list_indicators()
        for name in indicator_names:
            info = get_indicator(name)
            if info is not None:
                indicator_infos[name] = info
        available_indicators = {name: info.function for name, info in indicator_infos.items()}
    else:
        if available_indicators is None:
            available_indicators = {}
        indicator_names = list(available_indicators.keys())

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

    for name in indicator_names:
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

                # Paramètres depuis indicator_ranges.toml (fallback settings_class)
                params = {}
                param_specs = get_indicator_param_specs(ind_name, indicator_ranges)
                info = indicator_infos.get(ind_name)
                if info is None and REGISTRY_AVAILABLE:
                    info = get_indicator(ind_name)

                allowed_params = None
                if info and info.settings_class and is_dataclass(info.settings_class):
                    allowed_params = {field.name for field in fields(info.settings_class)}

                if param_specs:
                    params.update(
                        _build_params_from_specs(
                            ind_name,
                            key,
                            param_specs,
                            allowed_params=allowed_params,
                        )
                    )

                if info and info.settings_class:
                    params.update(
                        _build_params_from_settings(
                            ind_name,
                            key,
                            info.settings_class,
                            exclude=set(params.keys()),
                        )
                    )

                # Calculer l'indicateur
                try:
                    if use_registry and REGISTRY_AVAILABLE:
                        result = calculate_indicator(ind_name, df, params)
                    else:
                        # Préparer les données pour les indicateurs custom
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

                    result = _normalize_indicator_result(ind_name, result)

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
