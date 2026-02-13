from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="filtered_bollinger_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "williams_r", "keltner", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "bollinger_period": 20, "bollinger_std_dev": 2, "keltner_atr_period": 10, "keltner_ema_period": 20, "keltner_multiplier": 1.5, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50, "williams_r_period": 14}

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
        signals.iloc[:warmup] = 0.0
        
        if len(df) <= warmup:
            return signals
        
        close = df['close'].values
        bb = indicators['bollinger']
        bb_lower = np.nan_to_num(bb['lower'])
        bb_middle = np.nan_to_num(bb['middle'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        keltner = indicators['keltner']
        keltner_lower = np.nan_to_num(keltner['lower'])
        atr = np.nan_to_num(indicators['atr'])
        
        for i in range(warmup, len(df)):
            if signals[i-1] == 0.0:
                if (close[i] <= bb_lower[i] and 
                    williams_r[i] < -80 and 
                    close[i] > keltner_lower[i]):
                    signals[i] = 1.0
            elif signals[i-1] == 1.0:
                if close[i] >= bb_middle[i] or williams_r[i] > -20:
                    signals[i] = 0.0
                else:
                    signals[i] = 1.0
        
        return signals