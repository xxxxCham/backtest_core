from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ARBUSDC_MA')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'adx', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_overbought': 70,
         'adx_oversold': 30,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=14,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'ma_length': ParameterSpec(
                name='ma_length',
                min_val=2,
                max_val=99,
                default=9,
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
        if "leverage" in params and isinstance(params["leverage"], (int, float)):
            leverage = params["leverage"]
        else:
            leverage = 1

        # Add logic here to compute the indicator and update the signals dataframe with it. This may include filtering out bars outside of specified volatility bands using Bollinger Bands or other technical indicators.
        signals.iloc[:warmup] = 0.0
        return signals
