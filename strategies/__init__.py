"""
Backtest Core - Strategies Package
==================================

Stratégies de trading modulaires.
"""

from .base import StrategyBase, StrategyResult, get_strategy, list_strategies
from .bollinger_atr import BollingerATRStrategy
from .bollinger_best_longe_3i import BollingerBestLonge3iStrategy
from .bollinger_best_short_3i import BollingerBestShort3iStrategy
from .ema_cross import EMACrossStrategy
from .indicators_mapping import (
    STRATEGY_INDICATORS_MAP,
    get_all_indicators,
    get_required_indicators,
    get_strategy_info,
)
from .macd_cross import MACDCrossStrategy
from .rsi_reversal import RSIReversalStrategy
from .scalp_ema_bb_rsi_labs import ScalpEmaBbRsiLabsStrategy
from .scalping_bollinger_vwap_atr import ScalpingBollingerVwapAtrStrategy

__all__ = [
    "StrategyBase",
    "StrategyResult",
    "get_strategy",
    "list_strategies",
    "BollingerATRStrategy",
    "BollingerBestLonge3iStrategy",
    "BollingerBestShort3iStrategy",
    "EMACrossStrategy",
    "MACDCrossStrategy",
    "RSIReversalStrategy",
    "ScalpEmaBbRsiLabsStrategy",
    "ScalpingBollingerVwapAtrStrategy",
    "get_required_indicators",
    "get_all_indicators",
    "get_strategy_info",
    "STRATEGY_INDICATORS_MAP",
]
