from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='fichierstrat')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'mult_atr_stop': ParameterSpec(
                name='mult_atr_stop',
                min_val=2.0,
                max_val=8.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'mult_atr_tp': ParameterSpec(
                name='mult_atr_tp',
                min_val=5.0,
                max_val=30.0,
                default=5.5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(indicators, bars):
            signals = pd.Series(0., index=bars.index, dtype=np.float64)

            # Long signal
            indicators['donchian']['middle'] = indicators['donchian']['middle']
            adx_adx = indicators['adx']['adx']  # Need to use 'adx' in the format {'adx': ..., 'plus_di': ...}
            for i, bar in bars.iterrows():
                if bar.close > indicators['donchian']['middle'][i] and adx_adx[i] < 25:   # LONG intent
                    signals[i] = 1.0

            # Short signal
            indicators['donchian']['upper'] = indicators['donchian']['upper']
            adx_adx = indicators['adx']['adx']  # Need to use 'adx' in the format {'adx': ..., 'plus_di': ...}
            for i, bar in bars.iterrows():
                if bar.close < indicators['donchian']['upper'][i] and adx_adx[i] < 25:   # SHORT intent
                    signals[i] = -1.0

            return signals
        return signals
