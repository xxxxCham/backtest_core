from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_rsi_reversal')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=7,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=2,
                max_val=5,
                default=3,
                param_type='float',
                step=0.1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=10,
                max_val=25,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        supertrend = indicators['supertrend']
        rsi = indicators['rsi']
        atr = indicators['atr']

        supertrend_dir = indicators['supertrend']["direction"]
        supertrend_prev_dir = np.roll(supertrend_dir, 1)

        # LONG: supertrend direction changes from downtrend to uptrend AND rsi crosses above 30 from below
        supertrend_long_cond = (supertrend_prev_dir < 0) & (supertrend_dir > 0)
        rsi_long_cond = (np.roll(rsi, 1) < 30) & (rsi >= 30)
        long_entries = supertrend_long_cond & rsi_long_cond

        # SHORT: supertrend direction changes from uptrend to downtrend AND rsi crosses below 70 from above
        supertrend_short_cond = (supertrend_prev_dir > 0) & (supertrend_dir < 0)
        rsi_short_cond = (np.roll(rsi, 1) > 70) & (rsi <= 70)
        short_entries = supertrend_short_cond & rsi_short_cond

        signals[long_entries] = 1.0
        signals[short_entries] = -1.0

        # ATR-based stop loss and take profit
        atr_value = atr
        entry_mask = (signals != 0) & (np.roll(signals, 1) == 0)
        entry_prices = df['close'][entry_mask]

        for idx in df.index[entry_mask]:
            if signals[idx] == 1.0:  # Long entry
                df.loc[idx, 'bb_stop_long'] = df.loc[idx, 'close'] - 2 * atr_value[idx]
                df.loc[idx, 'bb_tp_long'] = df.loc[idx, 'close'] + 4 * atr_value[idx]
            elif signals[idx] == -1.0:  # Short entry
                df.loc[idx, 'bb_stop_short'] = df.loc[idx, 'close'] + 2 * atr_value[idx]
                df.loc[idx, 'bb_tp_short'] = df.loc[idx, 'close'] - 4 * atr_value[idx]
        signals.iloc[:warmup] = 0.0
        return signals
