from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_macd_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 75,
            "rsi_oversold": 35,
            "rsi_long_low": 35,
            "rsi_long_high": 75,
            "rsi_short_low": 30,
            "rsi_short_high": 60,
            "rsi_exit_overbought": 80,
            "rsi_exit_oversold": 20,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 4.5,
            "warmup": 35,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period",
                min_val=5,
                max_val=20,
                default=9,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=6.0,
                default=4.5,
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

        # Warm‑up
        warmup = int(params.get("warmup", 35))

        # Extract indicator arrays
        macd_dict = indicators['macd']
        macd_arr = np.nan_to_num(macd_dict["macd"])
        signal_arr = np.nan_to_num(macd_dict["signal"])
        hist_arr = np.nan_to_num(macd_dict["histogram"])
        rsi_arr = np.nan_to_num(indicators['rsi'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        # Cross detection
        prev_macd = np.roll(macd_arr, 1)
        prev_signal = np.roll(signal_arr, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_arr > signal_arr) & (prev_macd <= prev_signal)
        cross_down = (macd_arr < signal_arr) & (prev_macd >= prev_signal)

        # RSI thresholds
        rsi_long_low = params.get("rsi_long_low", params.get("rsi_oversold", 35))
        rsi_long_high = params.get("rsi_long_high", params.get("rsi_overbought", 75))
        rsi_short_low = params.get("rsi_short_low", 30)
        rsi_short_high = params.get("rsi_short_high", 60)
        rsi_exit_overbought = params.get("rsi_exit_overbought", 80)
        rsi_exit_oversold = params.get("rsi_exit_oversold", 20)

        # Long / short entry masks
        long_mask = (
            cross_up & (rsi_arr > rsi_long_low) & (rsi_arr < rsi_long_high)
        )
        short_mask = (
            cross_down & (rsi_arr > rsi_short_low) & (rsi_arr < rsi_short_high)
        )

        # Exit mask
        prev_hist = np.roll(hist_arr, 1)
        prev_hist[0] = np.nan
        hist_sign_change = (hist_arr * prev_hist < 0) & (~np.isnan(prev_hist))
        exit_mask = (
            hist_sign_change
            | (rsi_arr > rsi_exit_overbought)
            | (rsi_arr < rsi_exit_oversold)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warm‑up protection
        signals.iloc[:warmup] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close_arr[long_mask] - params["stop_atr_mult"] * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close_arr[long_mask] + params["tp_atr_mult"] * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close_arr[short_mask] + params["stop_atr_mult"] * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close_arr[short_mask] - params["tp_atr_mult"] * atr_arr[short_mask]

        return signals