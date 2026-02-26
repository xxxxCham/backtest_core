from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ATR-based Snake Breakout Strategy with ADR, OBV and 80% trend filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adn_period': 14,
         'adx_multiplier': 2.5,
         'adx_threshold': 20,
         'leverage': 1,
         'obv_median_period': 5,
         'stop_adr_mult': 3.0,
         'stop_atr_mult': 1.5,
         'tp_adr_mult': 7.0,
         'tp_atr_mult': 3.0,
         'warmup': 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adn_period': ParameterSpec(
                name='adn_period',
                min_val=8,
                max_val=40,
                default=14,
                param_type='int',
                step=1,
            ),
            'obv_median_period': ParameterSpec(
                name='obv_median_period',
                min_val=3,
                max_val=50,
                default=5,
                param_type='int',
                step=1,
            ),
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
        # This code is just a placeholder and should be replaced with the actual logic for generating buy/sell signals based on ADR, OBV, and 80% trend filter rules.
        long_mask = np.zeros(len(df), dtype=bool)  
        short_mask = np.zeros(len(df), dtype=bool)   

        # calculate moving average for each window size to use as a trailing stop loss and take profit level. 
        atr = indicators['atr']
        signals.iloc[:warmup] = 0.0
        return signals
