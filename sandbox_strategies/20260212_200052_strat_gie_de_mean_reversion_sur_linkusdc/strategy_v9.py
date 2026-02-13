from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 2, "stoch_rsi_overbought": 80, "stoch_rsi_oversold": 20, "stoch_rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(int, 10, 50, 1),
            "bollinger_std_dev": ParameterSpec(float, 1.0, 3.0, 0.1),
            "stoch_rsi_overbought": ParameterSpec(int, 70, 90, 1),
            "stoch_rsi_oversold": ParameterSpec(int, 10, 30, 1),
            "stoch_rsi_period": ParameterSpec(int, 10, 20, 1),
            "stop_atr_mult": ParameterSpec(float, 1.0, 2.5, 0.1),
            "tp_atr_mult": ParameterSpec(float, 2.0, 4.0, 0.1),
            "warmup": ParameterSpec(int, 30, 100, 1)
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
        
        if len(df) <= warmup:
            return signals
        
        close = df['close'].values
        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb['upper'])
        bb_middle = np.nan_to_num(bb['middle'])
        bb_lower = np.nan_to_num(bb['lower'])
        
        stoch_rsi = indicators['stoch_rsi']
        stoch_k = np.nan_to_num(stoch_rsi['k'])
        stoch_d = np.nan_to_num(stoch_rsi['d'])
        
        atr = np.nan_to_num(indicators['atr'])
        
        stoch_oversold = params.get("stoch_rsi_oversold", 20)
        stoch_overbought = params.get("stoch_rsi_overbought", 80)
        
        long_entries = (
            (close <= bb_lower) & 
            (stoch_k < stoch_oversold) & 
            (stoch_d < stoch_oversold)
        )
        
        short_entries = (
            (close >= bb_upper) & 
            (stoch_k > stoch_overbought) & 
            (stoch_d > stoch_overbought)
        )
        
        long_exits = close > bb_middle
        short_exits = close < bb_middle
        
        position = 0
        for i in range(warmup, len(signals)):
            if position == 0:
                if long_entries[i]:
                    signals.iloc[i] = 1.0
                    position = 1
                elif short_entries[i]:
                    signals.iloc[i] = -1.0
                    position = -1
                else:
                    signals.iloc[i] = 0.0
            elif position == 1:
                if long_exits[i]:
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    signals.iloc[i] = 1.0
            elif position == -1:
                if short_exits[i]:
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    signals.iloc[i] = -1.0
        
        return signals