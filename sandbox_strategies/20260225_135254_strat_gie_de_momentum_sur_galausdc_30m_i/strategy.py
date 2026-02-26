from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mfi_macd_atr_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['mfi', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'mfi_overbought': 70,
         'mfi_oversold': 30,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 2.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'mfi_overbought': ParameterSpec(
                name='mfi_overbought',
                min_val=60,
                max_val=80,
                default=70,
                param_type='int',
                step=1,
            ),
            'mfi_oversold': ParameterSpec(
                name='mfi_oversold',
                min_val=20,
                max_val=40,
                default=30,
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
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=2.9,
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
        indicators['macd']['macd'] = indicators['macd']['macd']
        signal_line = indicators['macd']['signal']
        mfi_arr = indicators['mfi']

        prev_macd = np.roll(indicators['macd']['macd'], 1)
        prev_signal = np.roll(signal_line, 1)
        # avoid wrap-around for first element
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan

        cross_above = (prev_macd <= prev_signal) & (indicators['macd']['macd'] > signal_line)
        cross_below = (prev_macd >= prev_signal) & (indicators['macd']['macd'] < signal_line)

        long_mask = cross_above & (mfi_arr > 70)
        short_mask = cross_below & (mfi_arr < 30)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
