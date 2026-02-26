from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_adx")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std": 2.7,
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 16,
            "stop_atr_mult": 1.25,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=50,
                default=16,
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
                min_val=1.5,
                max_val=3.5,
                default=2.7,
                param_type="float",
                step=0.1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.25,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
                max_val=10.0,
                default=5.5,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Masks initialized to avoid uninitialized variable errors
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Preserve warm‑up period
        signals.iloc[:warmup] = 0.0

        close = df["close"].values

        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])

        rsi = np.nan_to_num(indicators['rsi'])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Entry logic: Bollinger extremes + RSI extremes + ADX > 25
        adx_threshold = 25  # constant used when not supplied in params
        long_mask = (
            (close > upper) & (rsi > params["rsi_overbought"]) & (adx_arr > adx_threshold)
        )
        short_mask = (
            (close < lower) & (rsi < params["rsi_oversold"]) & (adx_arr > adx_threshold)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare stop‑loss and take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = long_mask
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[
            entry_long_mask
        ]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[
            entry_long_mask
        ]

        entry_short_mask = short_mask
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[
            entry_short_mask
        ]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[
            entry_short_mask
        ]

        signals.iloc[:warmup] = 0.0
        return signals