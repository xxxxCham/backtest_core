from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="stoch_macd_adx_atr_30m_audusdc")

    @property
    def required_indicators(self) -> List[str]:
        # ATR is needed for risk management
        return ["stochastic", "macd", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "stop_atr_mult": 1.3,
            "tp_atr_mult": 3.6,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=1.3,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=6.0,
                default=3.6,
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

        warmup = int(params.get("warmup", 50))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.6))

        # Extract indicator arrays
        stoch_k = np.nan_to_num(indicators['stochastic']["stoch_k"])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper for cross detection
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Entry conditions
        long_entry = (stoch_k < 20) & (macd_hist > 0) & (adx > 25)
        short_entry = (stoch_k > 80) & (macd_hist < 0) & (adx > 25)

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Exit conditions
        zero_hist = np.zeros_like(macd_hist)
        long_exit = (stoch_k > 80) | cross_down(macd_hist, zero_hist)
        short_exit = (stoch_k < 20) | cross_up(macd_hist, zero_hist)

        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP levels for entry bars only
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - (
            stop_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + (
            tp_atr_mult * atr[entry_long_mask]
        )

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + (
            stop_atr_mult * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - (
            tp_atr_mult * atr[entry_short_mask]
        )

        return signals