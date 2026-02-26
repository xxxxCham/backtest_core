from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="multi_factor_obv_supertrend_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["obv", "supertrend", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.0,
            "tp_atr_mult": 1.7,
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
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=3.0,
                default=1.7,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get("warmup", 20))

        # Wrap indicator arrays
        obv = np.nan_to_num(indicators['obv'])
        supertrend = np.nan_to_num(indicators['supertrend']["supertrend"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Trend of OBV
        obv_trend_up = obv > np.roll(obv, 1)
        obv_trend_down = obv < np.roll(obv, 1)

        # Entry conditions
        long_mask = obv_trend_up & (close > supertrend) & (rsi > 50)
        short_mask = obv_trend_down & (close < supertrend) & (rsi < 50)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Helper for previous values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_supertrend = np.roll(supertrend, 1)
        prev_supertrend[0] = np.nan
        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan

        # Exit conditions
        exit_long = (
            obv_trend_down
            | ((close < supertrend) & (prev_close >= prev_supertrend))
            | ((rsi < 50) & (prev_rsi >= 50))
        )
        exit_short = (
            obv_trend_up
            | ((close > supertrend) & (prev_close <= prev_supertrend))
            | ((rsi > 50) & (prev_rsi <= 50))
        )

        # Apply exits only when in position
        signals[exit_long & (signals == 1.0)] = 0.0
        signals[exit_short & (signals == -1.0)] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns for entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 1.7))

        long_entry_mask = signals == 1.0
        short_entry_mask = signals == -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = (
            close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        )
        df.loc[long_entry_mask, "bb_tp_long"] = (
            close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]
        )

        df.loc[short_entry_mask, "bb_stop_short"] = (
            close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        )
        df.loc[short_entry_mask, "bb_tp_short"] = (
            close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        )

        return signals