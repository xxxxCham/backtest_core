from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="breakout_donchian_adx")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 19,
            "adx_threshold": 25,
            "leverage": 1,
            "stop_atr_mult": 2.5,
            "tp_atr_mult": 4.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_threshold": ParameterSpec(
                name="adx_threshold",
                min_val=10,
                max_val=40,
                default=25,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=6.0,
                default=4.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Preserve warmup period
        signals.iloc[:warmup] = 0.0

        # Indicator arrays
        donch = indicators['donchian']
        upper = np.nan_to_num(donch["upper"])
        lower = np.nan_to_num(donch["lower"])
        middle = np.nan_to_num(donch["middle"])

        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr_arr = np.nan_to_num(indicators['atr'])

        close = df["close"].values

        # Entry conditions
        adx_thr = params.get("adx_threshold", 25)
        long_mask = (close > upper) & (adx_arr > adx_thr)
        short_mask = (close < lower) & (adx_arr > adx_thr)

        # Exit conditions: cross middle or ADX falls below threshold
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan

        cross_up = (close > middle) & (prev_close <= prev_middle)
        cross_down = (close < middle) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down
        exit_mask = cross_any | (adx_arr < adx_thr)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Optional flat on exit
        # signals[exit_mask] = 0.0

        # ATR‑based stop / take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - (
            params["stop_atr_mult"] * atr_arr[long_mask]
        )
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + (
            params["tp_atr_mult"] * atr_arr[long_mask]
        )
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + (
            params["stop_atr_mult"] * atr_arr[short_mask]
        )
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - (
            params["tp_atr_mult"] * atr_arr[short_mask]
        )

        signals.iloc[:warmup] = 0.0
        return signals