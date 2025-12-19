"""
Backtest Core - Strategies Package
==================================

Stratégies de trading modulaires.
"""

from .atr_channel import ATRChannelStrategy
from .base import StrategyBase, StrategyResult, get_strategy, list_strategies
from .bollinger_atr import BollingerATRStrategy
from .bollinger_dual import BollingerDualStrategy
from .ema_cross import EMACrossStrategy
from .ema_stochastic_scalp import EMAStochasticScalpStrategy
from .ma_crossover import MACrossoverStrategy
from .macd_cross import MACDCrossStrategy
from .rsi_reversal import RSIReversalStrategy
from .rsi_trend_filtered import RSITrendFilteredStrategy
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
    "BollingerDualStrategy",
    "EMACrossStrategy",
    "MACDCrossStrategy",
    "RSIReversalStrategy",
    "ATRChannelStrategy",
    "MACrossoverStrategy",
    "EMAStochasticScalpStrategy",
    "RSITrendFilteredStrategy",
    "get_required_indicators",
    "get_all_indicators",
    "get_strategy_info",
    "STRATEGY_INDICATORS_MAP",
]
