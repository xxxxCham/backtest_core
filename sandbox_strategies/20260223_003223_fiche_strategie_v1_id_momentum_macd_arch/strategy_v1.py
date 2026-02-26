from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_rsi_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'leverage': 1,
         'macd_fast': 8,
         'macd_signal': 10,
         'macd_slow': 32,
         'rsi_period': 20,
         'stop_atr_mult': 2.25,
         'tp_atr_mult': 4.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=20,
                default=8,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=20,
                max_val=50,
                default=32,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
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
                default=4.5,
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
        # Extract indicator arrays
        indicators['macd']['macd'] = indicators['macd']['macd']
        indicators['macd']['signal'] = indicators['macd']['signal']
        rsi = indicators['rsi']
        adx = indicators['adx']['adx']

        # Detect MACD crossovers
        macd_cross_above = (indicators['macd']['macd'] > indicators['macd']['signal']) & (np.roll(indicators['macd']['macd'], 1) <= np.roll(indicators['macd']['signal'], 1))
        macd_cross_below = (indicators['macd']['macd'] < indicators['macd']['signal']) & (np.roll(indicators['macd']['macd'], 1) >= np.roll(indicators['macd']['signal'], 1))

        # Build long and short masks with RSI and ADX conditions
        long_mask = macd_cross_above & (rsi > 45) & (rsi < 80) & (adx > 25)
        short_mask = macd_cross_below & (rsi > 30) & (rsi < 60) & (adx > 25)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
