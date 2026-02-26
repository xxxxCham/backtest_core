from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock\n(RSI+ATR+Stochastic)')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 6.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.0,
                max_val=8.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(self, df):
            signals = pd.Series(0.0, index=df.index) # Initialize signal series

            for rsi_period in [14]:  # Set period for RSI calculation
                for stochastic_window in [5]:  # Set window for Stochastic Oscillator calculation
                    for atr_period in [14]:  
                        signals += self.calculate_rsi(df, rsi_period) * self.calculate_stochastic(df, stochastic_window) * self.calculate_atr(df, atr_period) # Multiply RSI, Stochastic Oscillator and ATR values with corresponding weights

            long_mask = signals > 0   # Define conditions for taking a long position (i.e., buy signal)
            short_mask = signals < 0   # Define conditions for taking a short position (i.e., sell signal)

            signals[long_mask] *= -1  # Reverse the buy signal to take a short position in case of a long signal
            return pd.Series(data=signals, index=df.index)  # Return generated signals as pandas Series
        signals.iloc[:warmup] = 0.0
        return signals
