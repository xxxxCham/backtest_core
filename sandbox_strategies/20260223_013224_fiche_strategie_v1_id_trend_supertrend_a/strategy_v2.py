from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_adx_trend")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "atr_period": 14,
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_fast_period": ParameterSpec(
                name="ema_fast_period",
                min_val=5,
                max_val=30,
                default=20,
                param_type="int",
                step=1,
            ),
            "ema_slow_period": ParameterSpec(
                name="ema_slow_period",
                min_val=20,
                max_val=100,
                default=50,
                param_type="int",
                step=1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=10,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=10,
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
                max_val=6.0,
                default=3.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Ensure warmup slice is flat
        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # Compute EMAs directly from close to avoid dict key issues
        ema_fast_period = int(params.get("ema_fast_period", 20))
        ema_slow_period = int(params.get("ema_slow_period", 50))

        ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False).mean().values
        ema_slow = df["close"].ewm(span=ema_slow_period, adjust=False).mean().values

        # Cross detection
        prev_fast = np.roll(ema_fast, 1)
        prev_slow = np.roll(ema_slow, 1)
        prev_fast[0] = np.nan
        prev_slow[0] = np.nan

        cross_up = (ema_fast > ema_slow) & (prev_fast <= prev_slow)
        cross_down = (ema_fast < ema_slow) & (prev_fast >= prev_slow)

        # Entry conditions
        long_mask = (close > ema_fast) & cross_up & (adx_val > 25)
        short_mask = (close < ema_fast) & cross_down & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = cross_down | (adx_val < 20)
        signals[exit_mask] = 0.0

        # Risk‑management columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]

        # Ensure warmup bars remain flat
        signals.iloc[:warmup] = 0.0
        return signals