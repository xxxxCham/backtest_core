from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_filtered_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "roc", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bb_squeeze_threshold": 0.02, "bollinger_period": 20, "bollinger_std_dev": 1.5, "roc_period": 5, "stoch_rsi_overbought": 75, "stoch_rsi_oversold": 25, "stoch_rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bb_squeeze_threshold": ParameterSpec(min_value=0.005, max_value=0.05, step=0.005),
            "bollinger_period": ParameterSpec(min_value=10, max_value=30, step=1),
            "bollinger_std_dev": ParameterSpec(min_value=1.0, max_value=2.5, step=0.1),
            "roc_period": ParameterSpec(min_value=3, max_value=10, step=1),
            "stoch_rsi_overbought": ParameterSpec(min_value=70, max_value=80, step=5),
            "stoch_rsi_oversold": ParameterSpec(min_value=20, max_value=30, step=5),
            "stoch_rsi_period": ParameterSpec(min_value=10, max_value=20, step=1),
            "stop_atr_mult": ParameterSpec(min_value=1.0, max_value=2.0, step=0.1),
            "tp_atr_mult": ParameterSpec(min_value=2.0, max_value=4.0, step=0.1),
            "warmup": ParameterSpec(min_value=30, max_value=100, step=10),
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
        
        close = df['close'].values
        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb['upper'])
        bb_middle = np.nan_to_num(bb['middle'])
        bb_lower = np.nan_to_num(bb['lower'])
        stoch_rsi_k = np.nan_to_num(indicators['stoch_rsi']['k'])
        roc = np.nan_to_num(indicators['roc'])
        atr = np.nan_to_num(indicators['atr'])
        
        bb_squeeze_threshold = params.get('bb_squeeze_threshold', 0.02)
        stoch_rsi_oversold = params.get('stoch_rsi_oversold', 25)
        stoch_rsi_overbought = params.get('stoch_rsi_overbought', 75)
        
        bb_squeeze = (bb_upper - bb_lower) / bb_middle < bb_squeeze_threshold
        
        long_entry = (
            (close <= bb_lower) & 
            (stoch_rsi_k < stoch_rsi_oversold) & 
            (roc > 0) & 
            bb_squeeze
        )
        
        long_exit = (close >= bb_middle) | (stoch_rsi_k > stoch_rsi_overbought)
        
        position = 0
        for i in range(warmup, len(signals)):
            if position == 0 and long_entry[i]:
                signals.iloc[i] = 1.0
                position = 1
            elif position == 1 and long_exit[i]:
                signals.iloc[i] = 0.0
                position = 0
            else:
                signals.iloc[i] = signals.iloc[i-1]
        
        return signals