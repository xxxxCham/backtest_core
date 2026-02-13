from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_breakout_atr_trail")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "atr_period": 14,
            "bollinger_period": 20,
            "bollinger_std": 2,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
            "warmup": 50,
            "k_sl": 0.0,
            "sl_percent": 1.5,
            "tp_percent": 3.5,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(
                desc="Period for ADX calculation", min_val=1, max_val=100, step=1
            ),
            "atr_period": ParameterSpec(
                desc="Period for ATR calculation", min_val=1, max_val=100, step=1
            ),
            "bollinger_period": ParameterSpec(
                desc="Period for Bollinger Bands", min_val=5, max_val=50, step=1
            ),
            "bollinger_std": ParameterSpec(
                desc="Standard deviation multiplier for Bollinger Bands", min_val=0.5, max_val=5.0, step=0.1
            ),
            "stop_atr_mult": ParameterSpec(
                desc="ATR multiplier for stop-loss", min_val=0.1, max_val=10.0, step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                desc="ATR multiplier for take-profit", min_val=0.1, max_val=10.0, step=0.1
            ),
            "warmup": ParameterSpec(
                desc="Warmup periods (no signal)", min_val=0, max_val=200, step=1
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params["warmup"])
        signals.iloc[:warmup] = 0.0

        if len(df) < warmup:
            return signals

        bollinger = indicators["bollinger"]
        adx = indicators["adx"]
        atr = indicators["atr"]

        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])
        adx_value = np.nan_to_num(adx["adx"])
        median_volume = df["volume"].rolling(window=len(df) // 2).mean().values
        close = df["close"].values
        atr_value = np.nan_to_num(atr)

        long_entry = (
            (close > upper_band) &
            (adx_value > 25) &
            (df["volume"] > median_volume)
        )

        short_entry = (
            (close < lower_band) &
            (adx_value > 25) &
            (df["volume"] > median_volume)
        )

        long_exit = (
            (close < lower_band) |
            (close > upper_band) |
            (adx_value < 25)
        )

        short_exit = (
            (close > upper_band) |
            (close < lower_band) |
            (adx_value < 25)
        )

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        atr_mask = ~np.isnan(atr_value)
        valid_mask = ~np.isnan(close) & atr_mask & ~np.isnan(median_volume)

        entry_mask_long = long_entry & valid_mask
        close_long = close[entry_mask_long]
        atr_at_entry = atr_value[entry_mask_long]
        df.loc[entry_mask_long, "bb_stop_long"] = close_long - params["stop_atr_mult"] * atr_at_entry
        df.loc[entry_mask_long, "bb_tp_long"] = close_long + params["tp_atr_mult"] * atr_at_entry

        entry_mask_short = short_entry & valid_mask
        close_short = close[entry_mask_short]
        atr_at_entry = atr_value[entry_mask_short]
        df.loc[entry_mask_short, "bb_stop_short"] = close_short + params["stop_atr_mult"] * atr_at_entry
        df.loc[entry_mask_short, "bb_tp_short"] = close_short - params["tp_atr_mult"] * atr_at_entry

        return signals