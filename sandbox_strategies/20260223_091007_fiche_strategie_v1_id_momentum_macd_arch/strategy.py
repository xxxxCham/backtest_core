from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_rsi_sma_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'rsi_overbought': 55,
         'rsi_oversold': 45,
         'rsi_period': 14,
         'sma_period': 20,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 4.0,
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
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=30,
                max_val=50,
                default=45,
                param_type='int',
                step=1,
            ),
            'rsi_overbought': ParameterSpec(
                name='rsi_overbought',
                min_val=50,
                max_val=70,
                default=55,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=20,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=3,
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.0,
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
        # MACD cross above
        macd_cross_up = (
            (indicators['macd']['macd'] > indicators['macd']['signal'])
            & (np.roll(indicators['macd']['macd'], 1) <= np.roll(indicators['macd']['signal'], 1))
        )

        # MACD cross below
        macd_cross_down = (
            (indicators['macd']['macd'] < indicators['macd']['signal'])
            & (np.roll(indicators['macd']['macd'], 1) >= np.roll(indicators['macd']['signal'], 1))
        )

        # RSI between 45 and 55
        rsi_between = (indicators['rsi'] > 45) & (indicators['rsi'] < 55)

        # Close above SMA20
        close_gt_sma20 = df['close'] > indicators['sma']

        # Close below SMA20
        close_lt_sma20 = df['close'] < indicators['sma']

        # Long signal
        long_mask = macd_cross_up & rsi_between & close_gt_sma20
        signals[long_mask] = 1.0

        # Short signal
        short_mask = macd_cross_down & rsi_between & close_lt_sma20
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
