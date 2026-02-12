from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger_scalp_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "ema_period": 21,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_period": ParameterSpec(name="ema_period", type=int, default=21, min=1, max=200),
            "rsi_overbought": ParameterSpec(name="rsi_overbought", type=float, default=70.0, min=50.0, max=100.0),
            "rsi_oversold": ParameterSpec(name="rsi_oversold", type=float, default=30.0, min=0.0, max=50.0),
            "rsi_period": ParameterSpec(name="rsi_period", type=int, default=14, min=1, max=50),
            "stop_atr_mult": ParameterSpec(name="stop_atr_mult", type=float, default=1.5, min=0.5, max=5.0),
            "tp_atr_mult": ParameterSpec(name="tp_atr_mult", type=float, default=3.0, min=0.5, max=10.0),
            "warmup": ParameterSpec(name="warmup", type=int, default=50, min=0, max=200),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # Prepare price and indicator arrays
        close = df["close"].values.astype(np.float64)

        ema = np.nan_to_num(indicators["ema"]).astype(np.float64)
        rsi = np.nan_to_num(indicators["rsi"]).astype(np.float64)
        atr = np.nan_to_num(indicators["atr"]).astype(np.float64)

        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"]).astype(np.float64)
        upper_bb = np.nan_to_num(bb["upper"]).astype(np.float64)

        # Parameters
        rsi_overbought = float(params.get("rsi_overbought", 70))
        rsi_oversold = float(params.get("rsi_oversold", 30))
        stop_mult = float(params.get("stop_atr_mult", 1.5))
        tp_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))

        # Output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        position = 0.0  # 1.0 = long, -1.0 = short, 0.0 = flat
        entry_price = np.nan

        # Iterate over bars (skip first bar because we need previous values)
        for i in range(1, len(close)):
            # Default: keep previous position
            new_position = position

            if position == 0.0:
                # ----- LONG ENTRY -----
                cross_up = (close[i - 1] <= ema[i - 1]) and (close[i] > ema[i])
                at_lower = close[i] <= lower_bb[i]
                rsi_cond_long = (rsi[i - 1] < rsi_oversold) and (rsi[i] > rsi_oversold)

                if cross_up and at_lower and rsi_cond_long:
                    new_position = 1.0
                    entry_price = close[i]

                # ----- SHORT ENTRY -----
                cross_down = (close[i - 1] >= ema[i - 1]) and (close[i] < ema[i])
                at_upper = close[i] >= upper_bb[i]
                rsi_cond_short = (rsi[i - 1] > rsi_overbought) and (rsi[i] < rsi_overbought)

                if cross_down and at_upper and rsi_cond_short:
                    new_position = -1.0
                    entry_price = close[i]

            elif position == 1.0:
                # ----- LONG EXIT -----
                tp_price = entry_price + tp_mult * atr[i]
                sl_price = entry_price - stop_mult * atr[i]

                exit_long = (close[i] >= upper_bb[i]) or (close[i] >= tp_price) or (close[i] <= sl_price)

                if exit_long:
                    new_position = 0.0
                    entry_price = np.nan

            elif position == -1.0:
                # ----- SHORT EXIT -----
                tp_price = entry_price - tp_mult * atr[i]
                sl_price = entry_price + stop_mult * atr[i]

                exit_short = (close[i] <= lower_bb[i]) or (close[i] <= tp_price) or (close[i] >= sl_price)

                if exit_short:
                    new_position = 0.0
                    entry_price = np.nan

            # Record signal for current bar
            signals.iloc[i] = new_position
            position = new_position

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        return signals