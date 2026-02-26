from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_cci_mfi_atr_low_vol")

    @property
    def required_indicators(self) -> List[str]:
        return ["cci", "mfi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
            "cci_period": 20,
            "leverage": 1,
            "mfi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 2.8,
            "warmup": 60,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(
                name="cci_period",
                min_val=10,
                max_val=50,
                default=20,
                param_type="int",
                step=1,
            ),
            "mfi_period": ParameterSpec(
                name="mfi_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.8,
                param_type="float",
                step=0.1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Prepare indicator arrays
        cci = np.nan_to_num(indicators['cci'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Volatility filter: ATR below its median
        atr_median = np.nanmedian(atr)

        # Entry masks
        long_mask = (cci <= -100) & (mfi <= 20) & (atr < atr_median)
        short_mask = (cci >= 100) & (mfi >= 80) & (atr < atr_median)

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Helper functions for cross detection
        def cross_up(x: np.ndarray, y: float | np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            if np.isscalar(y):
                return (x > y) & (prev_x <= y)
            prev_y = np.roll(y, 1)
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: float | np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            if np.isscalar(y):
                return (x < y) & (prev_x >= y)
            prev_y = np.roll(y, 1)
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Exit masks: cross of CCI with 0 or MFI with 50
        cci_cross = cross_up(cci, 0.0) | cross_down(cci, 0.0)
        mfi_cross = cross_up(mfi, 50.0) | cross_down(mfi, 50.0)
        exit_mask = cci_cross | mfi_cross
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.8)

        # Long entry levels
        long_entry = signals == 1.0
        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        # Short entry levels
        short_entry = signals == -1.0
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        signals.iloc[:warmup] = 0.0
        return signals