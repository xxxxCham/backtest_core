from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_momentum_rsi_macd_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "macd", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.5, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(4.0, 6.0, 0.1),
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
        
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        rsi_cross_up = (rsi > rsi_oversold) & (rsi > np.roll(rsi, 1))
        macd_cross_up = (macd_histogram > 0) & (np.roll(macd_histogram, 1) <= 0)
        entry_condition = rsi_cross_up & macd_cross_up
        
        # Exit conditions
        rsi_cross_down = (rsi < rsi_overbought) & (rsi < np.roll(rsi, 1))
        rsi_below_50 = rsi < 50
        exit_condition = rsi_cross_down & rsi_below_50
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        for i in entry_indices:
            if i > warmup:
                signals.iloc[i] = 1.0  # LONG entry
                
                # Set stop-loss and take-profit levels
                sl = df["close"].iloc[i] - stop_atr_mult * atr[i]
                tp = df["close"].iloc[i] + tp_atr_mult * atr[i]
                
                # Apply exit logic
                for j in range(i + 1, len(signals)):
                    if df["close"].iloc[j] <= sl or df["close"].iloc[j] >= tp:
                        signals.iloc[j] = 0.0  # FLAT
                        break
                    if j in exit_indices:
                        signals.iloc[j] = 0.0  # FLAT
                        break
                        
        return signals