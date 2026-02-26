from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="lineausdc_30m_trend_ema_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "leverage": 1,
            "rsi_period": 14,
            "stop_atr_mult": 1.8,
            "tp_atr_mult": 3.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=14, param_type="int", step=1
            ),
            "ema_fast_period": ParameterSpec(
                name="ema_fast_period", min_val=5, max_val=100, default=20, param_type="int", step=1
            ),
            "ema_slow_period": ParameterSpec(
                name="ema_slow_period", min_val=20, max_val=200, default=50, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=1.8, param_type="float", step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=1.0, max_val=6.0, default=3.5, param_type="float", step=0.1
            ),
            "warmup": ParameterSpec(
                name="warmup", min_val=20, max_val=100, default=50, param_type="int", step=1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Parameters
        warmup = int(params.get("warmup", 50))
        ema_fast_period = int(params.get("ema_fast_period", 20))
        ema_slow_period = int(params.get("ema_slow_period", 50))

        # Close prices
        close = df["close"].values

        # Compute EMAs locally (since only one ema indicator is required by the framework)
        ema_fast = pd.Series(close).ewm(span=ema_fast_period, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=ema_slow_period, adjust=False).mean().values

        # RSI from indicators
        rsi = np.array(indicators.get('rsi', []))

        # Previous bar values
        prev_close = np.roll(close, 1)
        prev_ema_fast = np.roll(ema_fast, 1)

        # Cross detection
        cross_above = (close > ema_fast) & (prev_close <= prev_ema_fast)
        cross_below = (close < ema_fast) & (prev_close >= prev_ema_fast)

        # Avoid wrap-around at the first bar
        cross_above[0] = False
        cross_below[0] = False

        # Long and short masks
        long_mask = cross_above & (ema_fast > ema_slow) & (rsi > 55)
        short_mask = cross_below & (ema_fast < ema_slow) & (rsi < 45)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out signals during warm‑up
        signals.iloc[:warmup] = 0.0
        return signals