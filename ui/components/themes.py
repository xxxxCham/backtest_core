"""
Themes & Persistence - Gestion des thèmes UI et persistance des paramètres.

Composant Phase 5.6 pour personnaliser l'apparence de l'interface
et sauvegarder les préférences utilisateur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ThemeMode(Enum):
    """Mode de thème."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class ColorPalette(Enum):
    """Palettes de couleurs prédéfinies."""
    DEFAULT = "default"
    OCEAN = "ocean"
    FOREST = "forest"
    SUNSET = "sunset"
    MONOCHROME = "monochrome"
    CYBERPUNK = "cyberpunk"


# Définitions des palettes
PALETTES: Dict[ColorPalette, Dict[str, str]] = {
    ColorPalette.DEFAULT: {
        "primary": "#2196f3",
        "secondary": "#ff9800",
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336",
        "info": "#00bcd4",
        "background": "#0e1117",
        "surface": "#1e2130",
        "text": "#fafafa",
        "text_secondary": "#b0b0b0",
        "chart_up": "#26a69a",
        "chart_down": "#ef5350",
    },
    ColorPalette.OCEAN: {
        "primary": "#0077b6",
        "secondary": "#00b4d8",
        "success": "#06d6a0",
        "warning": "#ffd166",
        "error": "#ef476f",
        "info": "#118ab2",
        "background": "#03045e",
        "surface": "#023e8a",
        "text": "#caf0f8",
        "text_secondary": "#90e0ef",
        "chart_up": "#06d6a0",
        "chart_down": "#ef476f",
    },
    ColorPalette.FOREST: {
        "primary": "#2d6a4f",
        "secondary": "#40916c",
        "success": "#52b788",
        "warning": "#e9c46a",
        "error": "#e76f51",
        "info": "#74c69d",
        "background": "#1b4332",
        "surface": "#2d6a4f",
        "text": "#d8f3dc",
        "text_secondary": "#95d5b2",
        "chart_up": "#52b788",
        "chart_down": "#e76f51",
    },
    ColorPalette.SUNSET: {
        "primary": "#ff6b6b",
        "secondary": "#feca57",
        "success": "#1dd1a1",
        "warning": "#feca57",
        "error": "#ee5a24",
        "info": "#54a0ff",
        "background": "#2c2c54",
        "surface": "#474787",
        "text": "#f5f6fa",
        "text_secondary": "#dcdde1",
        "chart_up": "#1dd1a1",
        "chart_down": "#ee5a24",
    },
    ColorPalette.MONOCHROME: {
        "primary": "#888888",
        "secondary": "#666666",
        "success": "#a0a0a0",
        "warning": "#c0c0c0",
        "error": "#505050",
        "info": "#707070",
        "background": "#1a1a1a",
        "surface": "#2a2a2a",
        "text": "#e0e0e0",
        "text_secondary": "#909090",
        "chart_up": "#a0a0a0",
        "chart_down": "#505050",
    },
    ColorPalette.CYBERPUNK: {
        "primary": "#00ffff",
        "secondary": "#ff00ff",
        "success": "#00ff00",
        "warning": "#ffff00",
        "error": "#ff0000",
        "info": "#00ffff",
        "background": "#0a0a0a",
        "surface": "#1a1a2e",
        "text": "#eaeaea",
        "text_secondary": "#00ffff",
        "chart_up": "#00ff00",
        "chart_down": "#ff00ff",
    },
}


@dataclass
class ChartSettings:
    """Paramètres des graphiques."""
    default_height: int = 600
    show_volume: bool = True
    show_grid: bool = True
    show_legend: bool = True
    range_slider: bool = False
    candlestick_style: str = "hollow"  # "hollow", "filled", "ohlc"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceSettings:
    """Paramètres de performance."""
    use_gpu: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefaultParams:
    """Paramètres par défaut pour le backtest."""
    initial_capital: float = 10000.0
    fees_bps: int = 10
    slippage_bps: int = 5
    default_strategy: str = "ema_cross"
    default_timeframe: str = "1h"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserPreferences:
    """Préférences utilisateur complètes."""
    # Apparence
    theme_mode: ThemeMode = ThemeMode.DARK
    color_palette: ColorPalette = ColorPalette.DEFAULT
    font_size: str = "medium"  # "small", "medium", "large"
    
    # Graphiques
    chart_settings: ChartSettings = field(default_factory=ChartSettings)
    
    # Performance
    performance_settings: PerformanceSettings = field(default_factory=PerformanceSettings)
    
    # Défauts backtest
    default_params: DefaultParams = field(default_factory=DefaultParams)
    
    # Favoris
    favorite_strategies: List[str] = field(default_factory=list)
    favorite_indicators: List[str] = field(default_factory=list)
    recent_data_files: List[str] = field(default_factory=list)
    
    # UI State
    sidebar_expanded: bool = True
    show_tips: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise en dictionnaire."""
        return {
            "theme_mode": self.theme_mode.value,
            "color_palette": self.color_palette.value,
            "font_size": self.font_size,
            "chart_settings": self.chart_settings.to_dict(),
            "performance_settings": self.performance_settings.to_dict(),
            "default_params": self.default_params.to_dict(),
            "favorite_strategies": self.favorite_strategies,
            "favorite_indicators": self.favorite_indicators,
            "recent_data_files": self.recent_data_files,
            "sidebar_expanded": self.sidebar_expanded,
            "show_tips": self.show_tips,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Charge depuis un dictionnaire."""
        return cls(
            theme_mode=ThemeMode(data.get("theme_mode", "dark")),
            color_palette=ColorPalette(data.get("color_palette", "default")),
            font_size=data.get("font_size", "medium"),
            chart_settings=ChartSettings(**data.get("chart_settings", {})),
            performance_settings=PerformanceSettings(**data.get("performance_settings", {})),
            default_params=DefaultParams(**data.get("default_params", {})),
            favorite_strategies=data.get("favorite_strategies", []),
            favorite_indicators=data.get("favorite_indicators", []),
            recent_data_files=data.get("recent_data_files", []),
            sidebar_expanded=data.get("sidebar_expanded", True),
            show_tips=data.get("show_tips", True),
        )
    
    def get_colors(self) -> Dict[str, str]:
        """Retourne les couleurs de la palette active."""
        return PALETTES.get(self.color_palette, PALETTES[ColorPalette.DEFAULT])


class PreferencesManager:
    """
    Gestionnaire de persistance des préférences.
    
    Sauvegarde et charge les préférences utilisateur
    depuis un fichier JSON local.
    """
    
    DEFAULT_PATH = Path.home() / ".backtest_core" / "preferences.json"
    
    def __init__(self, path: Optional[Path] = None):
        """
        Initialise le gestionnaire.
        
        Args:
            path: Chemin du fichier de préférences
        """
        self.path = path or self.DEFAULT_PATH
        self._preferences: Optional[UserPreferences] = None
    
    @property
    def preferences(self) -> UserPreferences:
        """Retourne les préférences (charge si nécessaire)."""
        if self._preferences is None:
            self._preferences = self.load()
        return self._preferences
    
    def load(self) -> UserPreferences:
        """
        Charge les préférences depuis le fichier.
        
        Returns:
            Préférences chargées ou défaut
        """
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserPreferences.from_dict(data)
            except Exception:
                pass
        return UserPreferences()
    
    def save(self, preferences: Optional[UserPreferences] = None) -> bool:
        """
        Sauvegarde les préférences dans le fichier.
        
        Args:
            preferences: Préférences à sauvegarder (ou celles en cache)
            
        Returns:
            True si succès
        """
        prefs = preferences or self._preferences or UserPreferences()
        
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(prefs.to_dict(), f, indent=2)
            self._preferences = prefs
            return True
        except Exception:
            return False
    
    def update(self, **kwargs) -> UserPreferences:
        """
        Met à jour des préférences spécifiques.
        
        Args:
            **kwargs: Attributs à mettre à jour
            
        Returns:
            Préférences mises à jour
        """
        prefs = self.preferences
        
        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        
        self.save(prefs)
        return prefs
    
    def reset(self) -> UserPreferences:
        """Réinitialise aux valeurs par défaut."""
        self._preferences = UserPreferences()
        self.save(self._preferences)
        return self._preferences
    
    def add_favorite_strategy(self, strategy: str) -> None:
        """Ajoute une stratégie aux favoris."""
        prefs = self.preferences
        if strategy not in prefs.favorite_strategies:
            prefs.favorite_strategies.append(strategy)
            self.save(prefs)
    
    def remove_favorite_strategy(self, strategy: str) -> None:
        """Retire une stratégie des favoris."""
        prefs = self.preferences
        if strategy in prefs.favorite_strategies:
            prefs.favorite_strategies.remove(strategy)
            self.save(prefs)
    
    def add_recent_file(self, filepath: str, max_recent: int = 10) -> None:
        """Ajoute un fichier récent."""
        prefs = self.preferences
        if filepath in prefs.recent_data_files:
            prefs.recent_data_files.remove(filepath)
        prefs.recent_data_files.insert(0, filepath)
        prefs.recent_data_files = prefs.recent_data_files[:max_recent]
        self.save(prefs)


# Instance globale
_preferences_manager: Optional[PreferencesManager] = None


def get_preferences_manager() -> PreferencesManager:
    """Retourne l'instance globale du gestionnaire."""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = PreferencesManager()
    return _preferences_manager


def get_preferences() -> UserPreferences:
    """Raccourci pour obtenir les préférences."""
    return get_preferences_manager().preferences


def apply_theme(preferences: Optional[UserPreferences] = None) -> None:
    """
    Applique le thème aux composants Streamlit.
    
    Note: Streamlit ne supporte pas le changement de thème programmatique
    complet, mais on peut styler certains éléments.
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    prefs = preferences or get_preferences()
    colors = prefs.get_colors()
    
    # CSS personnalisé
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {colors['background']};
        }}
        .stMetric {{
            background-color: {colors['surface']};
            padding: 10px;
            border-radius: 5px;
        }}
        .stMetric label {{
            color: {colors['text_secondary']} !important;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            color: {colors['text']} !important;
        }}
        .css-1d391kg {{
            background-color: {colors['surface']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_settings(key: str = "theme_settings") -> Optional[UserPreferences]:
    """
    Rendu Streamlit des paramètres de thème.
    
    Args:
        key: Clé unique pour les widgets
        
    Returns:
        Préférences mises à jour ou None
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
    manager = get_preferences_manager()
    prefs = manager.preferences
    changed = False
    
    st.subheader("🎨 Thème et Apparence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mode thème
        theme_options = [m.value for m in ThemeMode]
        current_theme = theme_options.index(prefs.theme_mode.value)
        new_theme = st.selectbox(
            "Mode",
            theme_options,
            index=current_theme,
            key=f"{key}_theme_mode",
        )
        if new_theme != prefs.theme_mode.value:
            prefs.theme_mode = ThemeMode(new_theme)
            changed = True
    
    with col2:
        # Palette de couleurs
        palette_options = [p.value for p in ColorPalette]
        current_palette = palette_options.index(prefs.color_palette.value)
        new_palette = st.selectbox(
            "Palette",
            palette_options,
            index=current_palette,
            key=f"{key}_palette",
        )
        if new_palette != prefs.color_palette.value:
            prefs.color_palette = ColorPalette(new_palette)
            changed = True
    
    # Prévisualisation des couleurs
    colors = prefs.get_colors()
    st.markdown("**Aperçu des couleurs:**")
    
    cols = st.columns(6)
    color_keys = ["primary", "secondary", "success", "warning", "error", "info"]
    for i, key_name in enumerate(color_keys):
        color = colors[key_name]
        cols[i].markdown(
            f"<div style='background-color:{color};width:50px;height:30px;"
            f"border-radius:5px;margin:auto;'></div>"
            f"<small style='display:block;text-align:center'>{key_name}</small>",
            unsafe_allow_html=True,
        )
    
    # Taille de police
    font_options = ["small", "medium", "large"]
    font_options.index(prefs.font_size)
    new_font = st.select_slider(
        "Taille de police",
        options=font_options,
        value=prefs.font_size,
        key=f"{key}_font",
    )
    if new_font != prefs.font_size:
        prefs.font_size = new_font
        changed = True
    
    if changed:
        manager.save(prefs)
        st.success("✅ Thème mis à jour!")
        apply_theme(prefs)
    
    return prefs if changed else None


def render_chart_settings(key: str = "chart_settings") -> Optional[ChartSettings]:
    """
    Rendu des paramètres de graphique.
    
    Args:
        key: Clé unique
        
    Returns:
        Settings mis à jour ou None
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
    manager = get_preferences_manager()
    prefs = manager.preferences
    settings = prefs.chart_settings
    changed = False
    
    st.subheader("📊 Paramètres Graphiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_height = st.slider(
            "Hauteur par défaut",
            300, 1000, settings.default_height, 50,
            key=f"{key}_height",
        )
        if new_height != settings.default_height:
            settings.default_height = new_height
            changed = True
        
        new_volume = st.checkbox(
            "Afficher volume",
            value=settings.show_volume,
            key=f"{key}_volume",
        )
        if new_volume != settings.show_volume:
            settings.show_volume = new_volume
            changed = True
    
    with col2:
        new_grid = st.checkbox(
            "Afficher grille",
            value=settings.show_grid,
            key=f"{key}_grid",
        )
        if new_grid != settings.show_grid:
            settings.show_grid = new_grid
            changed = True
        
        new_legend = st.checkbox(
            "Afficher légende",
            value=settings.show_legend,
            key=f"{key}_legend",
        )
        if new_legend != settings.show_legend:
            settings.show_legend = new_legend
            changed = True
        
        new_slider = st.checkbox(
            "Range slider",
            value=settings.range_slider,
            key=f"{key}_slider",
        )
        if new_slider != settings.range_slider:
            settings.range_slider = new_slider
            changed = True
    
    # Style candlestick
    style_options = ["hollow", "filled", "ohlc"]
    current_style = style_options.index(settings.candlestick_style)
    new_style = st.selectbox(
        "Style chandelier",
        style_options,
        index=current_style,
        key=f"{key}_style",
    )
    if new_style != settings.candlestick_style:
        settings.candlestick_style = new_style
        changed = True
    
    if changed:
        manager.save(prefs)
        st.success("✅ Paramètres graphiques sauvegardés!")
    
    return settings if changed else None


def render_default_params(key: str = "default_params") -> Optional[DefaultParams]:
    """
    Rendu des paramètres par défaut de backtest.
    
    Args:
        key: Clé unique
        
    Returns:
        Params mis à jour ou None
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
    manager = get_preferences_manager()
    prefs = manager.preferences
    params = prefs.default_params
    changed = False
    
    st.subheader("⚙️ Paramètres par Défaut")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_capital = st.number_input(
            "Capital initial",
            min_value=100.0,
            max_value=10000000.0,
            value=params.initial_capital,
            step=1000.0,
            key=f"{key}_capital",
        )
        if new_capital != params.initial_capital:
            params.initial_capital = new_capital
            changed = True
        
        new_fees = st.number_input(
            "Frais (BPS)",
            min_value=0,
            max_value=100,
            value=params.fees_bps,
            step=1,
            key=f"{key}_fees",
        )
        if new_fees != params.fees_bps:
            params.fees_bps = new_fees
            changed = True
    
    with col2:
        new_slippage = st.number_input(
            "Slippage (BPS)",
            min_value=0,
            max_value=50,
            value=params.slippage_bps,
            step=1,
            key=f"{key}_slippage",
        )
        if new_slippage != params.slippage_bps:
            params.slippage_bps = new_slippage
            changed = True
        
        new_tf = st.selectbox(
            "Timeframe par défaut",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(params.default_timeframe),
            key=f"{key}_tf",
        )
        if new_tf != params.default_timeframe:
            params.default_timeframe = new_tf
            changed = True
    
    if changed:
        manager.save(prefs)
        st.success("✅ Paramètres par défaut sauvegardés!")
    
    return params if changed else None


def render_full_settings_page(key: str = "settings") -> None:
    """
    Page complète de paramètres.
    
    Args:
        key: Clé unique
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.title("⚙️ Paramètres")
    
    manager = get_preferences_manager()
    
    tabs = st.tabs(["🎨 Thème", "📊 Graphiques", "⚙️ Défauts", "⭐ Favoris"])
    
    with tabs[0]:
        render_theme_settings(f"{key}_theme")
    
    with tabs[1]:
        render_chart_settings(f"{key}_chart")
    
    with tabs[2]:
        render_default_params(f"{key}_params")
    
    with tabs[3]:
        prefs = manager.preferences
        
        st.subheader("⭐ Stratégies Favorites")
        if prefs.favorite_strategies:
            for strat in prefs.favorite_strategies:
                col1, col2 = st.columns([4, 1])
                col1.write(f"📈 {strat}")
                if col2.button("❌", key=f"{key}_remove_{strat}"):
                    manager.remove_favorite_strategy(strat)
                    st.rerun()
        else:
            st.info("Aucune stratégie favorite")
        
        st.subheader("📂 Fichiers Récents")
        if prefs.recent_data_files:
            for filepath in prefs.recent_data_files[:5]:
                st.caption(f"📄 {filepath}")
        else:
            st.info("Aucun fichier récent")
    
    # Actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Réinitialiser tout", key=f"{key}_reset"):
            manager.reset()
            st.success("✅ Préférences réinitialisées")
            st.rerun()
    
    with col2:
        prefs_json = json.dumps(manager.preferences.to_dict(), indent=2)
        st.download_button(
            "📥 Exporter",
            prefs_json,
            "backtest_preferences.json",
            "application/json",
            key=f"{key}_export",
        )
    
    with col3:
        uploaded = st.file_uploader(
            "📤 Importer",
            type="json",
            key=f"{key}_import",
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                data = json.load(uploaded)
                new_prefs = UserPreferences.from_dict(data)
                manager.save(new_prefs)
                st.success("✅ Préférences importées")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
