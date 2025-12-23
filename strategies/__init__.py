"""
Backtest Core - Strategies Package
==================================

Stratégies de trading modulaires.
"""

from .atr_channel import ATRChannelStrategy
from .base import StrategyBase, StrategyResult, get_strategy, list_strategies
from .bollinger_atr import BollingerATRStrategy
from .ema_cross import EMACrossStrategy
from .macd_cross import MACDCrossStrategy
from .rsi_reversal import RSIReversalStrategy
from .indicators_mapping import (
    get_required_indicators,
    get_all_indicators,
    get_strategy_info,
    STRATEGY_INDICATORS_MAP,
)

__all__ = [
    "StrategyBase",
    "StrategyResult",
    "get_strategy",
    "list_strategies",
    "BollingerATRStrategy",
    "EMACrossStrategy",
    "MACDCrossStrategy",
    "RSIReversalStrategy",
    "ATRChannelStrategy",
    "get_required_indicators",
    "get_all_indicators",
    "get_strategy_info",
    "STRATEGY_INDICATORS_MAP",
]
