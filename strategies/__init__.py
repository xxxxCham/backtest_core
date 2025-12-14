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
]
