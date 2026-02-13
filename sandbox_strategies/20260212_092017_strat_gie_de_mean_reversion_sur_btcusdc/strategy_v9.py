from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_btcusdc_30m")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 60, 80),
            "rsi_oversold": ParameterSpec(30, 20, 40),
            "rsi_period": ParameterSpec(14, 10, 20),
            "stop_atr_mult": ParameterSpec(1.5, 1.0, 2.0),
            "tp_atr_mult": ParameterSpec(3.0, 2.0, 4.0),
            "warmup": ParameterSpec(50, 30, 70),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        upper_bb = np.nan_to_num(bb["upper"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        atr_values = np.nan_to_num(atr)
        
        # Short only
        # Entry: price below lower BB AND stoch RSI crosses below 20
        price = np.nan_to_num(df["close"].values)
        prev_price = np.roll(price, 1)
        prev_stoch_rsi_k = np.roll(stoch_rsi_k, 1)
        prev_stoch_rsi_d = np.roll(stoch_rsi_d, 1)
        
        entry_cond = (price < lower_bb) & (prev_price >= lower_bb)
        stoch_cross_cond = (stoch_rsi_k < 20) & (prev_stoch_rsi_k >= 20)
        
        entry_mask = entry_cond & stoch_cross_cond
        
        # Exit: price crosses above middle BB
        exit_cond = (price > middle_bb) & (prev_price <= middle_bb)
        exit_mask = exit_cond
        
        # Set signals
        entry_indices = np.where(entry_mask)[0]
        exit_indices = np.where(exit_mask)[0]
        
        for i in entry_indices:
            signals.iloc[i] = -1.0  # SHORT
            
        for i in exit_indices:
            if signals.iloc[i] == -1.0:  # Only exit if currently short
                signals.iloc[i] = 0.0   # FLAT
                
        return signals