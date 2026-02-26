from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50,
         '},  ': {'default': 1.5, 'max': 4.0, 'min': 0.5, 'type': 'float'}}

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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df):
            # Define default parameters (leverage = 1)
            default_params = {'lmbda': 1}

            # Define a function to calculate ATR stop loss levels for long positions
            def atr_stop_long(bar):
                close = bar['close']
                atr = indicators['atr'][bar.name]
                return close - atr

            # Define a function to calculate ATR take profit levels for short positions
            def atr_tp_short(bar):
                close = bar['close']
                atr = indicators['atr'][bar.name]
                return close + atr

            # Define a function to calculate Bollinger band stop loss levels for long positions
            def bollinger_stop_long(bar):
                upper, middle, lower = [df[i][bar.name] for i in ['bollinger', 'middle']]
                return bar['close'] - 2 * (bar['close'] - lower)

            # Define a function to calculate Bollinger band take profit levels for short positions
            def bollinger_tp_short(bar):
                upper, middle, lower = [df[i][bar.name] for i in ['bollinger', 'middle']]
                return bar['close'] + 2 * (upper - bar['close'])

            # Define a function to calculate ADX stop loss levels for long positions
            def adx_stop_long(bar):
                plus = df[df.adx > 0]['plus_di'][bar.name]
                minus = df[df.adx < 0]['minus_di'][bar.name]
                return abs(2 * (plus - minus)) + 1e-6 # add a small constant to avoid division by zero errors

            # Define a function to calculate ADX take profit levels for short positions
            def adx_tp_short(bar):
                plus = df[df.adx > 0]['plus_di'][bar.name]
                minus = df[df.adx < 0]['minus_di'][bar.name]
                return abs(2 * (plus + minus)) - 1e-6 # add a small constant to avoid division by zero errors

            # Define the strategies for generating signals based on Bollinger bands, ADX and ATR indicators
        return signals
