from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=1,
                max_val=60,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
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
        def generate_signals(self, df, indicators, params):
            signals = pd.Series(0.0, index=df.index)

            for name, ind in indicators.items():
                if 'macd' in name or 'stochastic' in name:  # Check for MACD and stochastic oscillator
                    macd_period = int(params['macd'])
                    slow_period = int(params['slow'])

                    short_window = np.arange(-macd_period // 2, macd_period//2 + 1)
                    long_window = np.arange(-slow_period // 2, slow_period//2 + 1)

                    self._calc_MACD(short_window, long_window, ind['short'], ind['long']) # call internal function to calculate MACD and signal line

                elif 'bollinger' in name:   # Check for Bollinger Bands
                    window = int(params['periods'])  # periods is a parameter of the Bollinger Bands strategy
                    self._calc_BollingerBands(window)    # call internal function to calculate middle band and upper/lower bands

                else:                           # handle other strategies here if necessary
                     signals[name] = ind.update(df['close'])  # update signal for this indicator

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
