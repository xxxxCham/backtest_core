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
            "atr_period": 14,
            "bollinger_period": 20,
            "bollinger_std": 2.5,
            "leverage": 1,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "rsi_period": 14,
            "stop_atr_mult": 1.25,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=50, default=14, param_type="int", step=1
            ),
            "bollinger_period": ParameterSpec(
                name="bollinger_period", min_val=10, max_val=30, default=20, param_type="int", step=1
            ),
            "bollinger_std": ParameterSpec(
                name="bollinger_std", min_val=1.5, max_val=3.5, default=2.5, param_type="float", step=0.1
            ),
            "atr_period": ParameterSpec(
                name="atr_period", min_val=5, max_val=30, default=14, param_type="int", step=1
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult", min_val=0.5, max_val=4.0, default=1.25, param_type="float", step=0.1
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult", min_val=1.5, max_val=6.0, default=3.0, param_type="float", step=0.1
            ),
            "leverage": ParameterSpec(
                name="leverage", min_val=1, max_val=2, default=1, param_type="int", step=1
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Helper to detect crossovers
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Entry signals
        long_mask = (close < lower) & (rsi < params["rsi_oversold"])
        short_mask = (close > upper) & (rsi > params["rsi_overbought"])
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit signals
        # Use a full array for the constant 50.0 threshold to keep shapes aligned
        rsi_threshold = np.full_like(rsi, 50.0)
        exit_mask = cross_any(close, middle) | cross_any(rsi, rsi_threshold)
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        long_entry = signals == 1.0
        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - params["stop_atr_mult"] * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + params["tp_atr_mult"] * atr[long_entry]

        # Short entry SL/TP
        short_entry = signals == -1.0
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + params["stop_atr_mult"] * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - params["tp_atr_mult"] * atr[short_entry]

        # Final warmup reset (redundant but kept for safety)
        signals.iloc[:warmup] = 0.0
        return signals