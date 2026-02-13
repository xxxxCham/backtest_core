from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period": 14, "rsi_threshold": 50, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(5, 30, 1),
            "rsi_threshold": ParameterSpec(30, 70, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
            "warmup": ParameterSpec(20, 100, 10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        macd = indicators["macd"]
        macd_histogram = np.nan_to_num(macd["histogram"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        rsi_threshold = params.get("rsi_threshold", 50)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 4.5)
        warmup = int(params.get("warmup", 50))
        
        # Remove iloc usage - directly assign to numpy array
        signals.values[:warmup] = 0.0
        
        rsi_prev = np.roll(rsi, 1)
        macd_hist_prev = np.roll(macd_histogram, 1)
        
        # Entry conditions
        long_entry = (rsi > rsi_threshold) & (rsi_prev <= rsi_threshold) & (macd_histogram > 0) & (macd_hist_prev <= 0)
        short_entry = (rsi < rsi_threshold) & (rsi_prev >= rsi_threshold) & (macd_histogram < 0) & (macd_hist_prev >= 0)
        
        # Exit conditions
        long_exit = (rsi < rsi_threshold) | ((macd_histogram < 0) & (macd_hist_prev >= 0))
        short_exit = (rsi > rsi_threshold) | ((macd_histogram > 0) & (macd_hist_prev <= 0))
        
        # Generate signals
        positions = np.zeros_like(rsi)
        in_long = False
        in_short = False
        entry_price = 0.0
        entry_atr = 0.0
        
        for i in range(warmup, len(rsi)):
            if not in_long and not in_short:
                if long_entry[i]:
                    positions[i] = 1.0
                    in_long = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                elif short_entry[i]:
                    positions[i] = -1.0
                    in_short = True
                    entry_price = close[i]
                    entry_atr = atr[i]
            else:
                if in_long:
                    if long_exit[i]:
                        positions[i] = 0.0
                        in_long = False
                    else:
                        # Check for take profit or stop loss
                        if close[i] >= entry_price + tp_atr_mult * entry_atr:
                            positions[i] = 0.0
                            in_long = False
                        elif close[i] <= entry_price - stop_atr_mult * entry_atr:
                            positions[i] = 0.0
                            in_long = False
                elif in_short:
                    if short_exit[i]:
                        positions[i] = 0.0
                        in_short = False
                    else:
                        # Check for take profit or stop loss
                        if close[i] <= entry_price - tp_atr_mult * entry_atr:
                            positions[i] = 0.0
                            in_short = False
                        elif close[i] >= entry_price + stop_atr_mult * entry_atr:
                            positions[i] = 0.0
                            in_short = False
        
        # Remove iloc usage - directly assign to numpy array
        signals.values[warmup:] = positions[warmup:]
        return signals