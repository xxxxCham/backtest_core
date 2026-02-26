from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='fichie_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 18.0,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Check if the long and short masks are not reversed 
        mask_long = (signals == 1).shift(1) < signals  # Reversed Long Signal Mask
        mask_short = (signals != 1).shift(1) > signals   # Reversed Short Signal Mask

        # Check if the entry is within the warmup period and all conditions are met 
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # If signals are not reversed and within warm up period then proceed with logic below for generating signals

        if (params['leverage'] == 1):  
            leverage = params["leverage"]

        else:  
            leverage = 2   

        adx, atr_value, donchian_value= [], [], []

        # Generating ATR and Donchian Channel values for the given period
        signals.iloc[:warmup] = 0.0
        return signals
