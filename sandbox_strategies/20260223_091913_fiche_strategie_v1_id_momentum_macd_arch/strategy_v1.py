from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="macd_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 35,
            "rsi_period": 7,
            "stop_atr_mult": 3.0,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=30,
                default=7,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
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
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get("warmup", 50))
        # Initialise boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Prepare indicator arrays
        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_vals = np.nan_to_num(indicators['macd']["signal"])
        histogram = np.nan_to_num(indicators['macd']["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # MACD cross logic
        prev_macd = np.roll(macd_vals, 1)
        prev_signal = np.roll(signal_vals, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan

        cross_up = (macd_vals > signal_vals) & (prev_macd <= prev_signal)
        cross_down = (macd_vals < signal_vals) & (prev_macd >= prev_signal)

        # Entry conditions using the provided RSI thresholds
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]

        long_mask = cross_up & (rsi > rsi_oversold) & (rsi < rsi_overbought)
        short_mask = cross_down & (rsi > rsi_oversold) & (rsi < rsi_overbought)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_hist = np.roll(histogram, 1)
        prev_hist[0] = np.nan
        hist_cross = (np.sign(histogram) != np.sign(prev_hist))
        exit_mask = hist_cross | (rsi > rsi_overbought) | (rsi < rsi_oversold)
        signals[exit_mask] = 0.0

        # ATR‑based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        )

        signals.iloc[:warmup] = 0.0
        return signals