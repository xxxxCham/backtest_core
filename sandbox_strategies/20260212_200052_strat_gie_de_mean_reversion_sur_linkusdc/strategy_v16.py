from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_momentum_filtered_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "momentum", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bb_squeeze_threshold": 0.02, "bollinger_period": 20, "bollinger_std_dev": 2, "momentum_period": 10, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bb_squeeze_threshold": ParameterSpec("bb_squeeze_threshold", "float", 0.005, 0.05, 0.001),
            "bollinger_period": ParameterSpec("bollinger_period", "int", 10, 50, 1),
            "bollinger_std_dev": ParameterSpec("bollinger_std_dev", "float", 1.5, 3.0, 0.1),
            "momentum_period": ParameterSpec("momentum_period", "int", 5, 20, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", "float", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", "float", 1.5, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", "int", 30, 100, 1)
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
            
        close = np.nan_to_num(df['close'].values)
        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb['upper'])
        bb_middle = np.nan_to_num(bb['middle'])
        bb_lower = np.nan_to_num(bb['lower'])
        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])
        
        bb_squeeze_threshold = params.get('bb_squeeze_threshold', 0.02)
        
        squeeze_condition = (bb_upper - bb_lower) < (bb_middle * bb_squeeze_threshold)
        entry_condition = (close <= bb_lower) & (momentum > 0) & squeeze_condition
        exit_condition = (close >= bb_middle) | (momentum < 0)
        
        position = 0
        stop_price = 0.0
        tp_price = 0.0
        
        for i in range(warmup, len(df)):
            if position == 0:
                if entry_condition[i]:
                    signals.iloc[i] = 1.0
                    position = 1
                    stop_price = close[i] - (atr[i] * params.get('stop_atr_mult', 1.5))
                    tp_price = close[i] + (atr[i] * params.get('tp_atr_mult', 3.0))
            elif position == 1:
                if exit_condition[i] or close[i] <= stop_price or close[i] >= tp_price:
                    signals.iloc[i] = 0.0
                    position = 0
                else:
                    signals.iloc[i] = 1.0
        
        return signals