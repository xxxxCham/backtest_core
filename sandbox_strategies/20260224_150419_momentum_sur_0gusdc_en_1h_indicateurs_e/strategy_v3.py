from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
        def generate_signals(df, default_params=None):
            # Check if inputs have correct data type & shape (index should be DateTime)
            assert isinstance(df, pd.DataFrame), 'Input must be a Pandas DataFrame'

            # Check for Bollinger Bands and Donchian Channels strategy specific indicators 
            for indicator in ['bollinger', 'donchian']:
                if not isinstance(default_params[indicator], dict):
                    raise ValueError('Default parameters should be of type dict')

                assert 'upper' in default_params[indicator] and \
                       'middle' in default_params[indicator] and \
                       'lower' in default_params[indicator], \
                        'Bollinger Bands require upper, middle, lower bands as input'

            # Check for ATR-based Stop Loss & Take Profit strategy specific indicators 
            assert isinstance(default_params['atr'], dict), \
                'ATR must be a dictionary with keys "fast" and/or "slow"'

            if 'fast' in default_params['atr']:
                assert 'length' in default_params['atr']['fast'],\
                    'Fast ATR requires length as input'

            # Check for SuperTrend strategy specific indicators 
            assert isinstance(default_params['supertrend'], dict), \
                'SuperTrend must be a dictionary with keys "direction" and/or "length"'

            if 'length' in default_params['supertrend']:
                assert 'fast' in default_params['supertrend']['length'],\
                    'Length for SuperTrend direction or length input required' 

            # Check for Stochastic strategy specific indicators 
            assert isinstance(default_params['stochastic'], dict), \
                'Stochastic must be a dictionary with keys "k" and/or "d"'

            if 'k' in default_params['stochastic']:
                assert 'length' in default_params['stochastic']['k'],\
                    'K for Stochastic k input required' 

            # Check other preconditions
            assert isinstance(df, pd.DataFrame), \
                'Input must be a Pandas DataFrame'

            if not all([isinstance(i, (pd.Timestamp, str)) for i in df.index]):
                raise ValueError('Index should contain datetime or string type values')

            # Check and adjust default parameters 
            if default_params is None:
                default_params = {}

            for indicator, params in default_params.items():
                assert isinstance(indicator, str),\
                    'Default parameter keys should be of str type'

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
