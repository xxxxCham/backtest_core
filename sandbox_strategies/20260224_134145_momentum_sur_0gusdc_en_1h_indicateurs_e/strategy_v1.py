from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_overbought': 70,
         'bollinger_oversold': 30,
         'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 6.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=14,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'bollinger_oversold': ParameterSpec(
                name='bollinger_oversold',
                min_val=20,
                max_val=80,
                default=30,
                param_type='float',
                step=0.1,
            ),
            'bollinger_overbought': ParameterSpec(
                name='bollinger_overbought',
                min_val=80,
                max_val=95,
                default=70,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.5,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=6.0,
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
        def generate_signals(df, indicators):
            # For each bar in the dataframe
            for i, row in df.iterrows():
                close = np.array([row['close']])

                # Generate Bollinger Bands based on upper and middle bands using numpy arrays of bollinger['upper|middle|lower'] 
                if 'bollinger' in indicators:
                    lower_band, middle_band, upper_band = [np.array(indicators[ind][ind]) for ind in ['lower', 'middle', 'upper']]

                    # Use np.roll to roll the previous band backwards by 1 bar (to create a donchian breakout)
                    prev_lower_band, prev_middle_band = [np.roll(bb, -1)[0].reshape(-1,) for bb in [lower_band, middle_band]]

                # Calculate Donchian Bands based on upper and lower bands using numpy arrays of donchian['np.nan_to_num(indicators['donchian']['upper'])|indicators['donchian']['middle']|donchian_low'] 
                if 'donchian' in indicators:
                    donchian_bands = [np.array(indicators[ind][ind]) for ind in ['lower', 'mid', 'upper']]

                # Calculate ATR using numpy arrays of atr['atr|atr_true_range'] 
                if 'atr' in indicators:
                    atr = np.array([row['close'], row['high'], row['low']]).T - \
                        np.array([row['open'], row['open'], row['open']]) + \
                          1e-5 # add a small number to avoid zero division error 

                # Calculate RSI using numpy arrays of rsi['rsi_value|price'|...]  
                if 'rsi' in indicators:
                    rsi = (np.array(row) - np.min(row)) / \
                           (np.max(row) - np.min(row)) # RSI calculation with numpy arrays

                # Calculate Stochastic using numpy arrays of stochastic['stoch_k|stoch_d'] 
                if 'stochastic' in indicators:
                    stoch = ((np.array(row) / row['close']) - np.min([50, max(10, np.max(row))])) * 100 # Stochastic calculation with numpy arrays


                signals[i] = close[-1]
        signals.iloc[:warmup] = 0.0
        return signals
