from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_breakout")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
            "bollinger_period": 20,
            "bollinger_std": 2,
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 4.8,
            "trailing_atr_mult": 2.3,
            "warmup": 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "rsi_overbought": ParameterSpec(
                name="rsi_overbought",
                min_val=60,
                max_val=80,
                default=70,
                param_type="int",
                step=1,
            ),
            "rsi_oversold": ParameterSpec(
                name="rsi_oversold",
                min_val=20,
                max_val=40,
                default=30,
                param_type="int",
                step=1,
            ),
            "bollinger_period": ParameterSpec(
                name="bollinger_period",
                min_val=10,
                max_val=50,
                default=20,
                param_type="int",
                step=1,
            ),
            "bollinger_std": ParameterSpec(
                name="bollinger_std",
                min_val=1,
                max_val=3,
                default=2,
                param_type="float",
                step=0.1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=50,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=4.8,
                param_type="float",
                step=0.1,
            ),
            "trailing_atr_mult": ParameterSpec(
                name="trailing_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.3,
                param_type="float",
                step=0.1,
            ),
            "warmup": ParameterSpec(
                name="warmup",
                min_val=10,
                max_val=100,
                default=20,
                param_type="int",
                step=1,
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
        warmup = int(params.get("warmup", 20))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])

        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        # Entry conditions
        long_mask = (close > upper) & (rsi < rsi_overbought) & (atr > 0.02 * close)
        short_mask = (close < lower) & (rsi > rsi_oversold) & (atr > 0.02 * close)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = (close < middle) | (rsi > rsi_overbought) | (rsi < rsi_oversold)
        signals[exit_mask] = 0.0

        # Deduplicate consecutive identical signals using numpy for alignment safety
        sig_arr = signals.values
        change_mask = np.concatenate([[True], sig_arr[1:] != sig_arr[:-1]])
        sig_arr[~change_mask] = 0.0
        signals = pd.Series(sig_arr, index=df.index, dtype=np.float64)

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based levels on entry bars
        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        return signals