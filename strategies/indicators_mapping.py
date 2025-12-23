"""
Backtest Core - Mapping Stratégies → Indicateurs
=================================================

Fichier de référence centralisé qui associe chaque stratégie à ses indicateurs.
Ce mapping est utilisé par l'UI pour charger automatiquement les bons
indicateurs.

Structure:
- required_indicators: Indicateurs chargés automatiquement par le moteur
- internal_indicators: Indicateurs calculés internement par la stratégie
- all_indicators: Tous les indicateurs utilisés (requis + internes)
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class StrategyIndicators:
    """Définition des indicateurs pour une stratégie."""

    name: str
    required_indicators: List[str]  # Chargés par le moteur
    internal_indicators: List[str]  # Calculés par la stratégie
    description: str
    ui_label: str = ""

    @property
    def all_indicators(self) -> Set[str]:
        """Tous les indicateurs utilisés (requis + internes)."""
        return set(self.required_indicators + self.internal_indicators)

    def display_label(self) -> str:
        """Libelle d'affichage pour l'UI."""
        return self.ui_label or self.name


# =============================================================================
# MAPPING COMPLET STRATÉGIES → INDICATEURS
# =============================================================================

STRATEGY_INDICATORS_MAP: Dict[str, StrategyIndicators] = {

    # 1. ATR Channel
    "atr_channel": StrategyIndicators(
        name="ATR Channel",
        ui_label="📏 ATR Channel (Breakout)",
        required_indicators=["atr", "ema"],
        # ATR pour canal, EMA fournie en externe
        internal_indicators=[],  # Canal calculé à partir de l'EMA + ATR
        description="Breakout sur canal ATR avec filtre EMA"
    ),

    # 2. EMA Cross
    "ema_cross": StrategyIndicators(
        name="EMA Cross",
        ui_label="📈 EMA Crossover (Trend Following)",
        required_indicators=[],
        internal_indicators=["ema"],  # EMA rapide/lente calculées internement
        description="Croisement EMA simple (Golden/Death Cross)"
    ),

    # 3. Bollinger ATR
    "bollinger_atr": StrategyIndicators(
        name="Bollinger ATR",
        ui_label="📉 Bollinger + ATR (Mean Reversion)",
        required_indicators=["bollinger", "atr"],
        internal_indicators=[],
        description="Mean-reversion Bollinger avec filtre volatilité ATR"
    ),

    # 4. MACD Cross
    "macd_cross": StrategyIndicators(
        name="MACD Cross",
        ui_label="📊 MACD Crossover (Momentum)",
        required_indicators=["macd"],
        internal_indicators=[],
        description="Croisement MACD avec ligne signal"
    ),

    # 5. RSI Reversal
    "rsi_reversal": StrategyIndicators(
        name="RSI Reversal",
        ui_label="🔄 RSI Reversal (Mean Reversion)",
        required_indicators=["rsi"],
        internal_indicators=[],
        description="Mean-reversion sur niveaux RSI (survente/surachat)"
    ),

    # 6. MA Crossover
    "ma_crossover": StrategyIndicators(
        name="MA Crossover",
        ui_label="📐 MA Crossover (SMA Trend)",
        required_indicators=[],
        internal_indicators=["sma"],
        description="Croisement SMA rapide/lente"
    ),

    # 7. EMA Stochastic Scalp
    "ema_stochastic_scalp": StrategyIndicators(
        name="EMA Stochastic Scalp",
        ui_label="⚡ EMA + Stochastic (Scalping)",
        required_indicators=["stochastic"],
        internal_indicators=["ema"],
        description="Scalping avec filtre EMA et timing Stochastic"
    ),

    # 8. Bollinger Dual
    "bollinger_dual": StrategyIndicators(
        name="Bollinger Dual",
        ui_label="📊 Bollinger Dual (Mean Reversion)",
        required_indicators=["bollinger"],
        internal_indicators=["sma", "ema"],
        description="Bollinger + franchissement MA"
    ),

    # 9. RSI Trend Filtered
    "rsi_trend_filtered": StrategyIndicators(
        name="RSI Trend Filtered",
        ui_label="🔄 RSI Trend Filtered (Mean Rev.)",
        required_indicators=["rsi"],
        internal_indicators=["ema"],
        description="RSI filtre par tendance EMA"
    ),
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_required_indicators(strategy_name: str) -> List[str]:
    """
    Retourne la liste des indicateurs requis pour une stratégie.

    Args:
        strategy_name: Nom de la stratégie (ex: "bollinger_atr")

    Returns:
        Liste des indicateurs requis (ex: ["bollinger", "atr"])

    Raises:
        KeyError: Si la stratégie n'existe pas
    """
    if strategy_name not in STRATEGY_INDICATORS_MAP:
        raise KeyError(
            f"Stratégie '{strategy_name}' inconnue. "
            f"Disponibles: {list(STRATEGY_INDICATORS_MAP.keys())}"
        )

    return STRATEGY_INDICATORS_MAP[strategy_name].required_indicators


def get_all_indicators(strategy_name: str) -> Set[str]:
    """
    Retourne tous les indicateurs utilisés par une stratégie.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Set de tous les indicateurs (requis + internes)
    """
    if strategy_name not in STRATEGY_INDICATORS_MAP:
        raise KeyError(f"Stratégie '{strategy_name}' inconnue")

    return STRATEGY_INDICATORS_MAP[strategy_name].all_indicators


def get_internal_indicators(strategy_name: str) -> List[str]:
    """
    Retourne les indicateurs calculés internement par une stratégie.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Liste des indicateurs internes
    """
    if strategy_name not in STRATEGY_INDICATORS_MAP:
        raise KeyError(f"Stratégie '{strategy_name}' inconnue")

    return STRATEGY_INDICATORS_MAP[strategy_name].internal_indicators


def list_strategies() -> List[str]:
    """Liste toutes les stratégies disponibles."""
    return list(STRATEGY_INDICATORS_MAP.keys())


def get_strategy_info(strategy_name: str) -> StrategyIndicators:
    """
    Retourne les informations complètes sur une stratégie.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        StrategyIndicators avec toutes les infos
    """
    if strategy_name not in STRATEGY_INDICATORS_MAP:
        raise KeyError(f"Stratégie '{strategy_name}' inconnue")

    return STRATEGY_INDICATORS_MAP[strategy_name]


def format_strategy_summary() -> str:
    """
    Génère un résumé formaté de toutes les stratégies et leurs indicateurs.

    Returns:
        Résumé en format texte
    """
    lines = ["=" * 80]
    lines.append("RÉFÉRENCE STRATÉGIES → INDICATEURS")
    lines.append("=" * 80)
    lines.append("")

    for strategy_name, info in STRATEGY_INDICATORS_MAP.items():
        lines.append(f"📊 {info.name} ({strategy_name})")
        lines.append(f"   Description: {info.description}")
        required = ", ".join(info.required_indicators) or "Aucun"
        internal = ", ".join(info.internal_indicators) or "Aucun"
        lines.append(f"   Requis:      {required}")
        lines.append(f"   Internes:    {internal}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_strategy_indicators(
    strategy_name: str,
    strategy_instance
) -> bool:
    """
    Valide que les indicateurs déclarés dans le mapping correspondent
    aux indicateurs requis de la stratégie.

    Args:
        strategy_name: Nom de la stratégie
        strategy_instance: Instance de la stratégie

    Returns:
        True si cohérent, False sinon
    """
    if strategy_name not in STRATEGY_INDICATORS_MAP:
        return False

    expected = set(STRATEGY_INDICATORS_MAP[strategy_name].required_indicators)
    actual = set(strategy_instance.required_indicators)

    return expected == actual


__all__ = [
    "StrategyIndicators",
    "STRATEGY_INDICATORS_MAP",
    "get_required_indicators",
    "get_all_indicators",
    "get_internal_indicators",
    "list_strategies",
    "get_strategy_info",
    "format_strategy_summary",
    "validate_strategy_indicators",
]
