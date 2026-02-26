from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_ema_atr_scalp_5m')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.42,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=15,
                max_val=40,
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
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.42,
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
        macd_vals = indicators['macd']['macd']
        macd_sig = indicators['macd']['signal']
        macd_hist = indicators['macd']['histogram']
        ema_vals = indicators['ema']

        prev_macd = np.roll(macd_vals, 1)
        prev_sig = np.roll(macd_sig, 1)

        long_mask = (macd_vals > macd_sig) & (prev_macd <= prev_sig) & (macd_hist > 0) & (df['close'].values > ema_vals)
        short_mask = (macd_vals < macd_sig) & (prev_macd >= prev_sig) & (macd_hist < 0) & (df['close'].values < ema_vals)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
