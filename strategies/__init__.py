"""
Backtest Core - Strategies Package
==================================

Stratégies de trading modulaires.
"""

from .base import StrategyBase, StrategyResult, get_strategy, list_strategies
from .bollinger_atr import BollingerATRStrategy
from .bollinger_atr_v2 import BollingerATRStrategyV2
from .bollinger_atr_v3 import BollingerATRStrategyV3
from .ema_cross import EMACrossStrategy
from .indicators_mapping import (
    STRATEGY_INDICATORS_MAP,
    get_all_indicators,
    get_required_indicators,
    get_strategy_info,
)
from .macd_cross import MACDCrossStrategy
from .rsi_reversal import RSIReversalStrategy

__all__ = [
    "StrategyBase",
    "StrategyResult",
    "get_strategy",
    "list_strategies",
    "BollingerATRStrategy",
    "BollingerATRStrategyV2",
    "BollingerATRStrategyV3",
    "EMACrossStrategy",
    "MACDCrossStrategy",
    "RSIReversalStrategy",
    "get_required_indicators",
    "get_all_indicators",
    "get_strategy_info",
    "STRATEGY_INDICATORS_MAP",
]
