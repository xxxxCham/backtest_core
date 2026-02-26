from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='AAVEUSDC_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'adx']

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
        def ema(data, length):
            """
            Simple Moving Average with a custom length.

            Parameters:
                data (numpy array): The historical price data to calculate the EMA from.
                length (int): Length of EMA.

            Returns: 
                numpy array: EMA values for each point in 'data'.
            """
            return pd.Series(pd.Series(data).ewm(span=length, min_periods=length-1).mean(), index=data.index)

        def aroon(data):
            """
            Calculate the AROON up and down indicators from price data.

            Parameters: 
                data (numpy array): The historical prices to calculate the AROON indicator for.

            Returns: 
                tuple of numpy arrays: Tuple containing Aroon Up and Aroon Down values respectively.
            """
            up, down = pd.Series(pd.Series([100]*len(data)).diff(), index=data.index), pd.Series(pd.Series([100]*len(data)).diff()).shift(-2) - 100

            return (up[-2:], down[:-2])

        def adx(data, length):
            """
            Calculate the ADX indicator from price data.

            Parameters: 
                data (numpy array): The historical prices to calculate the ADX for.
                length (int): Length of the Donchian band.

            Returns: 
                numpy arrays: Upper, Middle and Lower ADX values respectively.
            """
            indicators['adx']['plus_di'] = pd.Series(pd.Series([0]*len(data)).diff(), index=data.index)
            indicators['adx']['minus_di'] = pd.Series(pd.Series([0]*len(data)).diff()).shift(-1) - 2 * (pd.Series(pd.Series([0]*len(data)).diff())).shift() + pd.Series(pd.Series([0]*len(data)).diff()).shift(-1)
            upper_band = up, down = pd.Series((indicators['adx']['plus_di']+indicators['adx']['minus_di'])/2., index=data.index), (down[:-3]+up[1:])/2.

            return upper_band  # TODO: Return DI values for lower and middle band as well.
        signals.iloc[:warmup] = 0.0
        return signals
