from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_min': 0.0005,
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 12,
            'stop_atr_mult': 2.5,
            'tp_atr_mult': 6.0,
            'warmup': 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=12,
                param_type='int',
                step=1,
            ),
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.1,
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
                min_val=2.0,
                max_val=10.0,
                default=6.0,
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
        params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Indicator arrays
        rsi = np.nan_to_num(indicators['rsi'])
        macd = np.nan_to_num(indicators['macd']['macd'])
        signal = np.nan_to_num(indicators['macd']['signal'])
        macd_hist = np.nan_to_num(indicators['macd']['histogram'])
        atr = np.nan_to_num(indicators['atr'])
        close = df['close'].values

        # Cross helpers
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd > signal) & (prev_macd <= prev_signal)
        cross_down = (macd < signal) & (prev_macd >= prev_signal)

        # Long and short entry conditions
        long_mask = (
            cross_up
            & (rsi > params['rsi_oversold'])
            & (rsi < params['rsi_overbought'])
            & (atr > params['atr_min'])
        )
        short_mask = (
            cross_down
            & (rsi > params['rsi_oversold'])
            & (rsi < params['rsi_overbought'])
            & (atr > params['atr_min'])
        )

        # Exit conditions
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        sign_change = (
            (macd_hist > 0) & (prev_hist <= 0) |
            (macd_hist < 0) & (prev_hist >= 0)
        )
        exit_mask = (
            sign_change
            | (rsi > params['rsi_overbought'])
            | (rsi < params['rsi_oversold'])
        )

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, 'bb_stop_long'] = np.nan
        df.loc[:, 'bb_tp_long'] = np.nan
        df.loc[:, 'bb_stop_short'] = np.nan
        df.loc[:, 'bb_tp_short'] = np.nan

        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, 'bb_stop_long'] = close[entry_long_mask] - params['stop_atr_mult'] * atr[entry_long_mask]
        df.loc[entry_long_mask, 'bb_tp_long'] = close[entry_long_mask] + params['tp_atr_mult'] * atr[entry_long_mask]

        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, 'bb_stop_short'] = close[entry_short_mask] + params['stop_atr_mult'] * atr[entry_short_mask]
        df.loc[entry_short_mask, 'bb_tp_short'] = close[entry_short_mask] - params['tp_atr_mult'] * atr[entry_short_mask]

        signals.iloc[:warmup] = 0.0
        return signals