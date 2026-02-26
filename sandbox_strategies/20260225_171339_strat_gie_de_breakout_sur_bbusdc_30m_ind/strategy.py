from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bbusdc_30m_supertrend_adx_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "supertrend_multiplier": 3.0,
            "supertrend_period": 10,
            "tp_atr_mult": 2.6,
            "warmup": 50,
            "adx_threshold": 25,  # added threshold for ADX
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "supertrend_period": ParameterSpec(
                name="supertrend_period",
                min_val=5,
                max_val=50,
                default=10,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type="float",
                step=0.1,
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
                default=2.6,
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
            "adx_threshold": ParameterSpec(
                name="adx_threshold",
                min_val=0,
                max_val=100,
                default=25,
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

        # Boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Wrap indicators
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        supertrend_level = np.nan_to_num(indicators['supertrend']["supertrend"])

        # Cross helpers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_st = np.roll(supertrend_level, 1)
        prev_st[0] = np.nan
        cross_up = (close > supertrend_level) & (prev_close <= prev_st)
        cross_down = (close < supertrend_level) & (prev_close >= prev_st)

        # Entry conditions
        adx_threshold = params.get("adx_threshold", 25)
        long_mask = cross_up & (adx_val > adx_threshold)
        short_mask = cross_down & (adx_val > adx_threshold)

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]

        signals.iloc[:warmup] = 0.0
        return signals