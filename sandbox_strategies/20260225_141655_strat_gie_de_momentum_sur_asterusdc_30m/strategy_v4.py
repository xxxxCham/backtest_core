from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_divergence_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'momentum_period': 10,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.1,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.1,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Calculate shifted arrays for vectorized comparisons
        momentum_prev = np.roll(indicators['momentum'], 1)
        macd = indicators['macd']['macd']
        indicators['macd']['signal'] = indicators['macd']['signal']
        macd_prev = np.roll(macd, 1)
        macd_signal_prev = np.roll(indicators['macd']['signal'], 1)
        low = df['low']
        high = df['high']
        low_prev = np.roll(low, 1)
        high_prev = np.roll(high, 1)

        # Momentum trend masks
        momentum_increasing = indicators['momentum'] > momentum_prev
        momentum_decreasing = indicators['momentum'] < momentum_prev

        # MACD crossover masks
        macd_cross_up = (macd > indicators['macd']['signal']) & (macd_prev <= macd_signal_prev)
        macd_cross_down = (macd < indicators['macd']['signal']) & (macd_prev >= macd_signal_prev)

        # Price low/high masks
        price_lower_low = low < low_prev
        price_higher_high = high > high_prev

        # Combine conditions for long and short signals
        long_mask = momentum_increasing & macd_cross_up & price_lower_low
        short_mask = momentum_decreasing & macd_cross_down & price_higher_high

        # Avoid signals on the first bar where roll wraps around
        long_mask[0] = False
        short_mask[0] = False

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
