"""
Backtest Core - Strategies Package.

Imports are best-effort to avoid hard failures when optional strategy files
are missing in a local workspace.
"""

from __future__ import annotations

from importlib import import_module

from .base import StrategyBase, StrategyResult, get_strategy, list_strategies
from .indicators_mapping import (
    STRATEGY_INDICATORS_MAP,
    get_all_indicators,
    get_required_indicators,
    get_strategy_info,
)

__all__ = [
    "StrategyBase",
    "StrategyResult",
    "get_strategy",
    "list_strategies",
    "get_required_indicators",
    "get_all_indicators",
    "get_strategy_info",
    "STRATEGY_INDICATORS_MAP",
]


def _optional_strategy(module_name: str, class_name: str) -> None:
    try:
        module = import_module(f".{module_name}", __name__)
        cls = getattr(module, class_name)
    except Exception:
        return
    globals()[class_name] = cls
    __all__.append(class_name)


_optional_strategy("bollinger_atr", "BollingerATRStrategy")
_optional_strategy("bollinger_best_longe_3i", "BollingerBestLonge3iStrategy")
_optional_strategy("bollinger_best_short_3i", "BollingerBestShort3iStrategy")
_optional_strategy("ema_cross", "EMACrossStrategy")
_optional_strategy("ema_rsi_regime", "EMARSIRegimeStrategy")
_optional_strategy("macd_cross", "MACDCrossStrategy")
_optional_strategy("rsi_reversal", "RSIReversalStrategy")
_optional_strategy("scalp_bb_vwap_rsi", "ScalpBollingerVwapRsiStrategy")
_optional_strategy("scalp_donchian_adx_breakout", "ScalpDonchianAdxBreakoutStrategy")
_optional_strategy("scalp_ema_rsi_pullback", "ScalpEmaRsiPullbackStrategy")
_optional_strategy("scalp_ema_bb_rsi_labs", "ScalpEmaBbRsiLabsStrategy")
_optional_strategy("scalping_bollinger_vwap_atr", "ScalpingBollingerVwapAtrStrategy")
