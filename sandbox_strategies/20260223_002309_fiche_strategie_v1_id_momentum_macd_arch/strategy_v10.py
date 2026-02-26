from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="macd_rsi_adx_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "rsi", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 18,
            "stop_atr_mult": 2.25,
            "tp_atr_mult": 5.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=18, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=2.25, param_type="float", step=0.1
            ),
            "adx_period": ParameterSpec(
                name="adx_period", min_val=5, max_val=50, default=14, param_type="int", step=1
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Unwrap indicator arrays
        macd_dict = indicators['macd']
        macd = np.nan_to_num(macd_dict["macd"])
        signal = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        adx = np.nan_to_num(indicators['adx']["adx"])

        # Cross helpers
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd > signal) & (prev_macd <= prev_signal)
        cross_down = (macd < signal) & (prev_macd >= prev_signal)

        # Histogram sign change
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        hist_sign = np.sign(hist)
        prev_sign = np.sign(prev_hist)
        hist_sign_change = (hist_sign != prev_sign) & (~np.isnan(prev_sign))

        # Entry and exit logic
        long_mask = cross_up & (rsi > 55) & (adx > 25)
        short_mask = cross_down & (rsi < 45) & (adx > 25)
        exit_mask = hist_sign_change | (rsi > params["rsi_overbought"]) | (rsi < params["rsi_oversold"])

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        close = df["close"].values

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        return signals