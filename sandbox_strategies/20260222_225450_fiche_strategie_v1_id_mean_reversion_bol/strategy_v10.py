from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 35,
            "rsi_period": 13,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 5.0,
            "warmup": 39,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=13, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=2.0, param_type="float", step=0.1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=2.0, max_val=4.5, default=5.0, param_type="float", step=0.1
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # unwrap indicators
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # entry conditions
        long_mask = (close < lower) & (rsi < params.get("rsi_oversold", 35))
        short_mask = (close > upper) & (rsi > params.get("rsi_overbought", 70))

        # exit conditions: cross over middle band or cross over RSI 50
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_rsi = np.roll(rsi, 1)

        # set first element to nan to avoid false cross detection
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        prev_rsi[0] = np.nan

        exit_mask = (
            ((close > middle) & (prev_close <= prev_middle))
            | ((close < middle) & (prev_close >= prev_middle))
            | ((rsi > 50) & (prev_rsi <= 50))
            | ((rsi < 50) & (prev_rsi >= 50))
        )

        # build signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels on entry bars
        stop_mult = params.get("stop_atr_mult", 2.0)
        tp_mult = params.get("tp_atr_mult", 5.0)

        # initialize columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals