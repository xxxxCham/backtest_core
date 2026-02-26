from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

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
        def generate_signals(indicators):
            # Assuming indicators is a dict with Bollinger Bands as keys ('upper'|'middle'|'lower') 
            donchian = {}   # Initialize empty dictionary for Donchian Channels
            adx, indicators['adx']['plus_di'], indicators['adx']['minus_di'] = {},{},{}  # Initialize empty dictionaries for ADX components
            supertrend = {'supertrend':1} if 'supertrend' in indicators else None     # Initialize empty dict for Supertrend direction (None if not available)
            stochastic = {}   # Initialize empty dictionary for Stochastic K and D values 
            atr_values = []   # Create an array to hold ATR values from indicator list

            # Iterate over Bollinger Bands, Donchian Channels, ADX components, Supertrend direction, and Stochastic K/D
            for band in ['upper','middle','lower']:
                donchian[band] = indicators.get(f'{band}_donchian', np.array([]))  # Initialize empty list for Donchian values if not available

                adx_val, plus_di_val, minus_di_val = {},{},{}   # Initialize empty dicts for ADX components (if available)

            # Calculate the ATR and STOCH K/D indicators. Ignore float scalar operations on these arrays.
            atr_values = [np.array(atr_value) if isinstance(atr_value, list) else atr_value 
                         for atr_value in indicators['atr']]   # Flatten nested lists or convert single values to array

            stochastic_k_d = {k: np.array(v) if k not in ('signal', 'stoch_k') and v is not None else None  # Convert dicts to arrays with NaN where missing data
                              for k, v in indicators['stochastic'].items()}   # Iterate over 'Stochastic K' and 'Stochastic D' values

            signal = signals.copy()   # Make a copy of the Series before modifying it
        signals.iloc[:warmup] = 0.0
        return signals
