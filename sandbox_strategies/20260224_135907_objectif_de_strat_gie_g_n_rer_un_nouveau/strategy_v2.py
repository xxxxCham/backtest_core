from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='CryptoTrendStrategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'macd', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 30,
         'leverage': 2,
         'macd_fast_length': 8,
         'macd_signal_length': 9,
         'macd_slow_length': 13,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'macd_fast_length': ParameterSpec(
                name='macd_fast_length',
                min_val=4,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_length': ParameterSpec(
                name='macd_slow_length',
                min_val=6,
                max_val=50,
                default=18,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=3,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(indicators, df):
            long_intent = {
                'bollinger': {'upper': 20},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'ema': {'short': 34, 'long': 58}
            }

            short_intent = {
                'bollinger': {'upper': -20},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'ema': {'short': 34, 'long': 58}
            }

            # Check each indicator's condition for long and short signals
            for name in indicators:
                if name not in ['bollinger', 'macd', 'ema']:
                    continue

                val = getattr(indicators[name], 'upper')  # Get the value from dict

                close_price = df['close'].values[-1]  # Last closing price

                upper, middle, lower = long_intent.get(name)

                if name == 'bollinger':
                    prev_band = np.roll(upper, -1)

                    if close_price > prev_band:
                        signals[i] = 1.0

            # Continue with other indicators' logic...
        signals.iloc[:warmup] = 0.0
        return signals
