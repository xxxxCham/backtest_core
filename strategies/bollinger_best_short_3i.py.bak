"""
Module-ID: strategies.bollinger_best_short_3i

Purpose: Bollinger level-based SHORT mirror strategy with entry/SL/TP on band scale.

Role in pipeline: trading strategy

Key components: BollingerBestShort3iStrategy, register_strategy("bollinger_best_short_3i")

Inputs: DataFrame OHLCV, parameters (bb_period, bb_std, entry_level, sl_level, tp_level, leverage)

Outputs: StrategyResult signals (-1/0), Bollinger levels, metadata

Dependencies: pandas, numpy, utils.parameters, strategies.base

Conventions: Scale 0.0=lower band, 0.5=middle, 1.0=upper. Entry near upper band (0.8 to 1.0).
Stop-loss above upper band (1.3 to 1.8). Take-profit toward lower band (0.0 to 0.3).

Read-if: Adjusting mirror ranges or entry logic.

Skip-if: Editing other strategies.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import SAFE_RANGES_PRESET, ParameterSpec, Preset

from .base import StrategyBase, register_strategy


@register_strategy("bollinger_best_short_3i")
class BollingerBestShort3iStrategy(StrategyBase):
    """
    Bollinger level-based SHORT mirror strategy.

    Scale reference:
        0.0 = lower_band
        0.5 = middle_band
        1.0 = upper_band

    Parameters:
        entry_level: 0.8 to 1.0 (near upper band)
        sl_level: 1.3 to 1.8 (above upper band)
        tp_level: 0.0 to 0.3 (toward lower band)
    """

    def __init__(self) -> None:
        super().__init__(name="Bollinger_best_short_3i")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.1,
            "entry_level": 1.0,
            "sl_level": 1.5,
            "tp_level": 0.15,
            "leverage": 1,
            "fees_bps": 10,
            "slippage_bps": 5,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bb_period": ParameterSpec(
                name="bb_period",
                min_val=10, max_val=50, default=20,
                param_type="int",
                description="Bollinger period",
            ),
            "bb_std": ParameterSpec(
                name="bb_std",
                min_val=1.0, max_val=4.0, step=0.1, default=2.1,
                param_type="float",
                description="Bollinger std dev",
            ),
            "entry_level": ParameterSpec(
                name="entry_level",
                min_val=0.8, max_val=1.0, step=0.05, default=1.0,
                param_type="float",
                description="Entry level near upper band (0.8 to 1.0)",
            ),
            "sl_level": ParameterSpec(
                name="sl_level",
                min_val=1.3, max_val=1.8, step=0.05, default=1.5,
                param_type="float",
                description="Stop-loss level above upper band",
            ),
            "tp_level": ParameterSpec(
                name="tp_level",
                min_val=0.0, max_val=0.3, step=0.05, default=0.15,
                param_type="float",
                description="Take-profit level toward lower band",
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1, max_val=10, default=1,
                param_type="int",
                description="Leverage (not optimized)",
                optimize=False,
            ),
        }

    def get_preset(self) -> Optional[Preset]:
        return SAFE_RANGES_PRESET

    def get_indicator_params(
        self,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if indicator_name == "bollinger":
            return {
                "period": int(params.get("bb_period", 20)),
                "std_dev": float(params.get("bb_std", 2.1)),
            }
        return super().get_indicator_params(indicator_name, params)

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64, name="signals")

        if "bollinger" not in indicators or indicators["bollinger"] is None:
            return signals

        bb_result = indicators["bollinger"]
        if not isinstance(bb_result, tuple) or len(bb_result) < 3:
            return signals

        upper, middle, lower = bb_result[:3]

        if not isinstance(upper, pd.Series):
            upper = pd.Series(np.asarray(upper), index=df.index)
        if not isinstance(lower, pd.Series):
            lower = pd.Series(np.asarray(lower), index=df.index)
        if not isinstance(middle, pd.Series):
            middle = pd.Series(np.asarray(middle), index=df.index)

        close = df["close"]

        entry_level = float(params.get("entry_level", 1.0))
        total_distance = upper - lower
        entry_price_level = lower + entry_level * total_distance

        short_condition = close >= entry_price_level
        signals[short_condition] = -1.0

        sl_level = float(params.get("sl_level", 1.5))
        tp_level = float(params.get("tp_level", 0.15))
        stop_short = lower + sl_level * total_distance
        tp_short = lower + tp_level * total_distance

        df.loc[:, "bb_entry_short"] = entry_price_level
        df.loc[:, "bb_stop_short"] = stop_short
        df.loc[:, "bb_tp_short"] = tp_short
        df.loc[:, "bb_upper"] = upper
        df.loc[:, "bb_middle"] = middle
        df.loc[:, "bb_lower"] = lower

        signals_diff = signals.diff()
        signals_clean = signals.copy()
        signals_clean[1:] = np.where(signals_diff[1:] != 0, signals[1:], 0)

        return signals_clean

    def _resolve_level_price(
        self,
        entry_price: float,
        atr_value: float,
        params: Dict[str, Any],
        level_key: str,
        bb_upper: Optional[float],
        bb_lower: Optional[float],
    ) -> float:
        entry_level = float(params.get("entry_level", 1.0))
        level = float(params.get(level_key, entry_level))

        if bb_upper is not None and bb_lower is not None:
            total_distance = bb_upper - bb_lower
            base = bb_lower
        else:
            total_distance = atr_value * 2.0 if atr_value else entry_price * 0.01
            base = entry_price - entry_level * total_distance

        if total_distance == 0:
            return entry_price

        return base + level * total_distance

    def get_stop_loss(
        self,
        entry_price: float,
        atr_value: float,
        side: str,
        params: Dict[str, Any],
        bb_middle: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
    ) -> float:
        _ = side
        return self._resolve_level_price(
            entry_price,
            atr_value,
            params,
            "sl_level",
            bb_upper,
            bb_lower,
        )

    def get_take_profit(
        self,
        entry_price: float,
        atr_value: float,
        side: str,
        params: Dict[str, Any],
        bb_middle: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
    ) -> float:
        _ = side
        return self._resolve_level_price(
            entry_price,
            atr_value,
            params,
            "tp_level",
            bb_upper,
            bb_lower,
        )


__all__ = ["BollingerBestShort3iStrategy"]