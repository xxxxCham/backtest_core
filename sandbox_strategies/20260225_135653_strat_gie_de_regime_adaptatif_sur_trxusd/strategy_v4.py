from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="atr_adx_regime_adaptive")

    @property
    def required_indicators(self) -> List[str]:
        return ["atr", "adx", "sma"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "adx_threshold": 25,
            "atr_period": 14,
            "atr_threshold": 1.5,
            "leverage": 1,
            "sma_period": 20,
            "stop_atr_mult": 2.5,
            "tp_atr_mult": 3.0,
            "tp_range_atr_mult": 1.6,
            "tp_trend_atr_mult": 4.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "sma_period": ParameterSpec(
                name="sma_period",
                min_val=10,
                max_val=50,
                default=20,
                param_type="int",
                step=1,
            ),
            "atr_threshold": ParameterSpec(
                name="atr_threshold",
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type="float",
                step=0.1,
            ),
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
            "tp_trend_atr_mult": ParameterSpec(
                name="tp_trend_atr_mult",
                min_val=2.0,
                max_val=6.0,
                default=4.0,
                param_type="float",
                step=0.1,
            ),
            "tp_range_atr_mult": ParameterSpec(
                name="tp_range_atr_mult",
                min_val=1.0,
                max_val=3.0,
                default=1.6,
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
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Prepare columns for stops and take‑profits
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        sma = np.nan_to_num(indicators['sma'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        long_mask = (
            (close > sma)
            & (adx > params["adx_threshold"])
            & (atr > params["atr_threshold"])
        )
        short_mask = (
            (close < sma)
            & (adx > params["adx_threshold"])
            & (atr > params["atr_threshold"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_trend_atr_mult"] * atr[long_mask]
        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_trend_atr_mult"] * atr[short_mask]

        prev_close = np.roll(close, 1)
        prev_sma = np.roll(sma, 1)
        prev_close[0] = np.nan
        prev_sma[0] = np.nan

        cross_down = (close < sma) & (prev_close >= prev_sma)
        cross_up = (close > sma) & (prev_close <= prev_sma)
        # Use a default exit threshold of 20 if not provided
        adx_exit_threshold = params.get("adx_exit_threshold", 20)
        adx_low = adx < adx_exit_threshold

        exit_mask = cross_down | cross_up | adx_low
        signals[exit_mask] = 0.0

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals