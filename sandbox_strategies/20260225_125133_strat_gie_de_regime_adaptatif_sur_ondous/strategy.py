from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_rsi_atr_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 50,
            'leverage': 1,
            'rsi_lower': 45,
            'rsi_period': 14,
            'rsi_upper': 55,
            'stop_atr_mult': 2.2,
            'tp_atr_mult': 4.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'rsi_upper': ParameterSpec(
                name='rsi_upper',
                min_val=50,
                max_val=80,
                default=55,
                param_type='int',
                step=1,
            ),
            'rsi_lower': ParameterSpec(
                name='rsi_lower',
                min_val=20,
                max_val=50,
                default=45,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.5,
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
        """Generate long (+1) / short (-1) signals.

        - Long entry:  close > EMA and RSI > rsi_upper
        - Short entry: close < EMA and RSI < rsi_lower
        - Warm‑up period is forced to neutral (0).
        """
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # warm‑up handling
        warmup = int(params.get('warmup', self.default_params['warmup']))

        # price series
        close_arr = df['close'].values

        # indicator arrays (already numpy)
        ema_arr = indicators['ema']
        rsi_arr = indicators['rsi']
        atr_arr = indicators['atr']

        # thresholds from params (fallback to defaults)
        rsi_upper = params.get('rsi_upper', self.default_params['rsi_upper'])
        rsi_lower = params.get('rsi_lower', self.default_params['rsi_lower'])
        stop_mult = params.get('stop_atr_mult', self.default_params['stop_atr_mult'])
        tp_mult = params.get('tp_atr_mult', self.default_params['tp_atr_mult'])

        # entry masks
        long_mask = (close_arr > ema_arr) & (rsi_arr > rsi_upper)
        short_mask = (close_arr < ema_arr) & (rsi_arr < rsi_lower)

        # assign entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # optional: expose stop / TP levels for downstream use
        # (stored on the dataframe; harmless if not used)
        stop_long = close_arr - stop_mult * atr_arr
        tp_long = close_arr + tp_mult * atr_arr
        stop_short = close_arr + stop_mult * atr_arr
        tp_short = close_arr - tp_mult * atr_arr

        df['bb_stop_long'] = np.where(long_mask, stop_long, np.nan)
        df['bb_tp_long'] = np.where(long_mask, tp_long, np.nan)
        df['bb_stop_short'] = np.where(short_mask, stop_short, np.nan)
        df['bb_tp_short'] = np.where(short_mask, tp_short, np.nan)

        # enforce warm‑up neutrality
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals