from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "ema_fast": 9,
            "ema_mid": 21,
            "ema_slow": 50,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_fast": ParameterSpec(default=9),
            "ema_mid": ParameterSpec(default=21),
            "ema_slow": ParameterSpec(default=50),
            "rsi_overbought": ParameterSpec(default=70),
            "rsi_oversold": ParameterSpec(default=30),
            "stop_atr_mult": ParameterSpec(default=1.5),
            "tp_atr_mult": ParameterSpec(default=3.0),
            "warmup": ParameterSpec(default=50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare indicator arrays with NaN handling
        ema = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])

        bb = indicators["bollinger"]
        lower_band = np.nan_to_num(bb["lower"])
        upper_band = np.nan_to_num(bb["upper"])

        # Extract parameters
        rsi_oversold = float(params.get("rsi_oversold", 30))
        rsi_overbought = float(params.get("rsi_overbought", 70))
        warmup = int(params.get("warmup", 50))

        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Position tracking
        position = 0.0  # 1.0 = long, -1.0 = short, 0.0 = flat

        close_series = df["close"].values

        for i in range(len(df)):
            if i < warmup:
                continue

            close = close_series[i]
            close_prev = close_series[i - 1] if i > 0 else close

            ema_cur = ema[i]
            ema_prev = ema[i - 1] if i > 0 else ema_cur

            rsi_cur = rsi[i]
            rsi_prev = rsi[i - 1] if i > 0 else rsi_cur

            lower = lower_band[i]
            upper = upper_band[i]

            # ENTRY LOGIC
            if position == 0.0:
                # Long entry: pull‑back bounce
                long_entry = (
                    (close_prev < ema_prev)
                    & (close > ema_cur)
                    & (rsi_prev < rsi_oversold)
                    & (rsi_cur > rsi_oversold)
                    & (close <= lower)
                )
                # Short entry: pull‑back bounce opposite direction
                short_entry = (
                    (close_prev > ema_prev)
                    & (close < ema_cur)
                    & (rsi_prev > rsi_overbought)
                    & (rsi_cur < rsi_overbought)
                    & (close >= upper)
                )
                if long_entry:
                    position = 1.0
                elif short_entry:
                    position = -1.0

            # EXIT LOGIC
            else:
                if position == 1.0:
                    # Take profit: reach opposite Bollinger band or cross EMA (using same EMA as proxy)
                    tp_condition = (close >= upper) | (close >= ema_cur)
                    # Early exit: price moves against EMA
                    early_exit = close < ema_cur
                    if tp_condition | early_exit:
                        position = 0.0
                elif position == -1.0:
                    tp_condition = (close <= lower) | (close <= ema_cur)
                    early_exit = close > ema_cur
                    if tp_condition | early_exit:
                        position = 0.0

            signals.iloc[i] = position

        return signals