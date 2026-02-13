from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "bollinger_period": 20, "bollinger_std_dev": 2, "stoch_rsi_d": 3, "stoch_rsi_k": 3, "stoch_rsi_rsi_period": 14, "stoch_rsi_stoch_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        
        if len(df) <= warmup:
            return signals
        
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = indicators["atr"]
        
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        atr_vals = np.nan_to_num(atr)
        close = np.nan_to_num(df["close"].values)
        
        long_entry = (close <= bb_lower) & (stoch_rsi_k < 20) & (stoch_rsi_d < 30)
        long_exit = (close >= bb_middle) | (stoch_rsi_k > 80)
        
        position = 0
        entry_price = 0.0
        
        for i in range(warmup, len(signals)):
            if position == 0 and long_entry[i]:
                signals.iloc[i] = 1.0
                position = 1
                entry_price = close[i]
            elif position == 1:
                if long_exit[i]:
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    stop_loss = entry_price - (atr_vals[i] * params["stop_atr_mult"])
                    take_profit = entry_price + (atr_vals[i] * params["tp_atr_mult"])
                    if close[i] <= stop_loss or close[i] >= take_profit:
                        signals.iloc[i] = 0.0
                        position = 0
                    else:
                        signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
        
        return signals