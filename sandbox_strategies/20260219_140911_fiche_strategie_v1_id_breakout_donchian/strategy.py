from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_cross_adx_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "ema_long_period": 50,
            "ema_short_period": 20,
            "leverage": 1,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_short_period": ParameterSpec(
                name="ema_short_period",
                min_val=5,
                max_val=50,
                default=20,
                param_type="int",
                step=1,
            ),
            "ema_long_period": ParameterSpec(
                name="ema_long_period",
                min_val=10,
                max_val=100,
                default=50,
                param_type="int",
                step=1,
            ),
            "adx_period": ParameterSpec(
                name="adx_period",
                min_val=5,
                max_val=20,
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
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Compute short and long EMAs from close price
        close = df["close"]
        ema_short = close.ewm(span=params["ema_short_period"], adjust=False).mean().to_numpy()
        ema_long = close.ewm(span=params["ema_long_period"], adjust=False).mean().to_numpy()

        # Cross detection using numpy shift
        cross_above = (ema_short > ema_long) & (np.roll(ema_short, 1) <= np.roll(ema_long, 1))
        cross_below = (ema_short < ema_long) & (np.roll(ema_short, 1) >= np.roll(ema_long, 1))

        # ADX filter
        adx_mask = indicators['adx']["adx"] > 25

        # Long and short signal masks
        long_mask = (close.to_numpy() > ema_short) & cross_above & adx_mask
        short_mask = (close.to_numpy() < ema_short) & cross_below & adx_mask

        # Build signals series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out warmup period
        signals.iloc[:warmup] = 0.0
        return signals