from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='avntusdc_scalp_ema_macd_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'ema_period': 20,
            'leverage': 1,
            'macd_fast': 12,
            'macd_signal': 9,
            'macd_slow': 26,
            'stop_atr_mult': 2.1,
            'tp_atr_mult': 4.2,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
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
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.2,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        close = df['close'].values
        ema = indicators['ema']  # already a numpy array
        macd = indicators['macd']['macd']
        signal_line = indicators['macd']['signal']

        long_mask = (close > ema) & (macd > signal_line)
        short_mask = (close < ema) & (macd < signal_line)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        atr = indicators['atr']
        stop_atr_mult = params.get('stop_atr_mult', 2.1)
        tp_atr_mult = params.get('tp_atr_mult', 4.2)

        df['bb_stop_long'] = np.where(long_mask, close - stop_atr_mult * atr, np.nan)
        df['bb_tp_long'] = np.where(long_mask, close + tp_atr_mult * atr, np.nan)
        df['bb_stop_short'] = np.where(short_mask, close + stop_atr_mult * atr, np.nan)
        df['bb_tp_short'] = np.where(short_mask, close - tp_atr_mult * atr, np.nan)

        # Zero out signals during warm‑up period
        signals[:warmup] = 0.0

        return signals