from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ["momentum", "roc"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_stop_mult": 1.5, "momentum_period": 12, "roc_period": 12, "tp_atr_mult": 3.0, "warmup": 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_stop_mult": ParameterSpec(type_=(float, int), default=1.5, min_=0.1, max_=5.0),
            "momentum_period": ParameterSpec(type_=(int,), default=12, min_=1, max_=50),
            "roc_period": ParameterSpec(type_=(int,), default=12, min_=1, max_=50),
            "tp_atr_mult": ParameterSpec(type_=(float, int), default=3.0, min_=0.1, max_=10.0),
            "warmup": ParameterSpec(type_=(int,), default=20, min_=0, max_=1000),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        momentum = np.nan_to_num(indicators["momentum"])
        roc = np.nan_to_num(indicators["roc"])
        
        warmup = int(params["warmup"])
        signals.iloc[:warmup] = 0.0
        
        for i in range(warmup, len(momentum)):
            current_momentum = momentum[i]
            prev_momentum = momentum[i-1]
            current_roc = roc[i]
            
            if current_roc < 0 and current_momentum < prev_momentum:
                signals.iloc[i] = -1.0
            elif current_roc > 0 and current_momentum > prev_momentum:
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
                
        return signals