from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Asymmetric Breakout with Asymmetry Risk Management')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'bollinger', 'volume_oscillator']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'ema_period': 14,
         'leverage': 1,
         'macd_fast_length': 9,
         'macd_slow_length': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volatility_std_dev': 2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=1,
                max_val=100,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_fast_length': ParameterSpec(
                name='macd_fast_length',
                min_val=2,
                max_val=99,
                default=9,
                param_type='int',
                step=1,
            ),
            'macd_slow_length': ParameterSpec(
                name='macd_slow_length',
                min_val=2,
                max_val=99,
                default=26,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'volatility_std_dev': ParameterSpec(
                name='volatility_std_dev',
                min_val=1,
                max_val=3,
                default=2,
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
        def generate_signals(self, df):
                # Compute necessary indicators here (e.g., Bollinger Bands and MACD)
                macd = MACD(df["close"])

                # Generate signals based on indicator values and custom logic
                signal = ta.MACDMAGMA(macd)  # Convert to a boolean signal based on threshold

                # Assign the generated signals back into the dataframe
                signals = signal

                return df
        signals.iloc[:warmup] = 0.0
        return signals
