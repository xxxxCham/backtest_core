from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_ema_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 2.5, 'warmup': 60}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "supertrend_atr_period": ParameterSpec(
                name="supertrend_atr_period",
                min_val=3,
                max_val=20,
                default=5,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1.0,
                max_val=4.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=30,
                default=15,
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
            "ema_period": ParameterSpec(
                name="ema_period",
                min_val=10,
                max_val=200,
                default=50,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=1.75,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # --- Prepare indicator arrays (all float) ---
        close = np.nan_to_num(df["close"].values, nan=0.0)
        ema = np.nan_to_num(indicators['ema'], nan=0.0)
        # supertrend direction may be int; cast to float to allow NaN
        direction = np.array(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.nan_to_num(indicators['adx']["adx"], nan=0.0)
        atr = np.nan_to_num(indicators['atr'], nan=0.0)

        # --- Entry masks ---
        long_mask = (direction == 1) & (adx_val > 30) & (close > ema)
        short_mask = (direction == -1) & (adx_val > 30) & (close < ema)

        # Avoid duplicate consecutive signals
        if n > 1:
            long_mask &= (signals.values != 1.0)
            short_mask &= (signals.values != -1.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # --- Exit logic: direction change or weak ADX ---
        prev_direction = np.roll(direction, 1).astype(float)
        prev_direction[0] = np.nan
        direction_change = direction != prev_direction
        adx_exit = adx_val < 20
        exit_mask = direction_change | adx_exit
        exit_mask &= (signals.values != 0.0)

        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # --- ATR-based SL/TP levels ---
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        return signals