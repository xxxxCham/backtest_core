from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_persistence')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.5, 'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.5,
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
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # --- Parameters ---
        warmup = int(params.get('warmup', 50))
        stop_atr_mult = params.get('stop_atr_mult', 1.5)
        tp_atr_mult = params.get('tp_atr_mult', 3.5)
        adx_threshold = params.get('adx_threshold', 25)
        direction_streak = int(params.get('direction_streak', 3))

        # --- Indicator arrays ---
        st_dir = np.array(indicators['supertrend']['direction'], dtype=float)
        adx_val = np.array(indicators['adx']['adx'], dtype=float)
        atr_arr = np.array(indicators['atr'], dtype=float)
        close = df['close'].values

        # --- Direction unchanged streak mask ---
        if direction_streak > 1:
            # Compute streak without introducing NaN into integer arrays
            unchanged = np.zeros(n, dtype=bool)
            if n >= direction_streak:
                # Compare current with previous two
                unchanged[direction_streak - 1 :] = (
                    (st_dir[direction_streak - 1 :] == st_dir[direction_streak - 2 : -1])
                    & (st_dir[direction_streak - 1 :] == st_dir[direction_streak - 3 : -2])
                )
        else:
            unchanged = np.ones(n, dtype=bool)

        # --- Entry masks ---
        long_mask = (st_dir == 1.0) & (adx_val > adx_threshold) & unchanged
        short_mask = (st_dir == -1.0) & (adx_val > adx_threshold) & unchanged

        # --- Avoid duplicate consecutive signals ---
        prev_sig = np.roll(signals.values, 1)
        prev_sig[0] = 0.0
        long_mask &= (prev_sig != 1.0)
        short_mask &= (prev_sig != -1.0)

        # --- Warmup period ---
        long_mask[:warmup] = False
        short_mask[:warmup] = False
        signals.iloc[:warmup] = 0.0

        # --- Assign signals ---
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # --- ATR-based SL/TP levels ---
        df.loc[:, 'bb_stop_long'] = np.nan
        df.loc[:, 'bb_tp_long'] = np.nan
        df.loc[:, 'bb_stop_short'] = np.nan
        df.loc[:, 'bb_tp_short'] = np.nan

        df.loc[long_mask, 'bb_stop_long'] = close[long_mask] - stop_atr_mult * atr_arr[long_mask]
        df.loc[long_mask, 'bb_tp_long'] = close[long_mask] + tp_atr_mult * atr_arr[long_mask]
        df.loc[short_mask, 'bb_stop_short'] = close[short_mask] + stop_atr_mult * atr_arr[short_mask]
        df.loc[short_mask, 'bb_tp_short'] = close[short_mask] - tp_atr_mult * atr_arr[short_mask]

        return signals