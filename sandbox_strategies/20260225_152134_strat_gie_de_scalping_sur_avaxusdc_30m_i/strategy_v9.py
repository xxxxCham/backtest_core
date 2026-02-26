from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_macd_atr_scalp')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.0,
         'warmup': 50}

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
                min_val=15,
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
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=50,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        close = df['close'].values
        ema20 = indicators['ema']
        macd_hist = indicators['macd']['histogram']
        indicators['macd']['macd'] = indicators['macd']['macd']
        macd_signal_line = indicators['macd']['signal']

        # Cross above / below EMA
        cross_above_ema = (close > ema20) & (np.roll(close, 1) <= np.roll(ema20, 1))
        cross_below_ema = (close < ema20) & (np.roll(close, 1) >= np.roll(ema20, 1))

        # Long and short conditions
        long_mask = cross_above_ema & (macd_hist > 0) & (indicators['macd']['macd'] > macd_signal_line)
        short_mask = cross_below_ema & (macd_hist < 0) & (indicators['macd']['macd'] < macd_signal_line)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop/TP on entry bars
        atr = indicators['atr']
        stop_long = df['close'] - 2 * atr
        tp_long = df['close'] + 2 * atr
        stop_short = df['close'] + 2 * atr
        tp_short = df['close'] - 2 * atr

        df.loc[signals == 1.0, 'bb_stop_long'] = stop_long[signals == 1.0]
        df.loc[signals == 1.0, 'bb_tp_long'] = tp_long[signals == 1.0]
        df.loc[signals == -1.0, 'bb_stop_short'] = stop_short[signals == -1.0]
        df.loc[signals == -1.0, 'bb_tp_short'] = tp_short[signals == -1.0]
        signals.iloc[:warmup] = 0.0
        return signals
