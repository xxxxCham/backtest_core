from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_supertrend_adx_atr_v5')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'atr_period': 14,
            'leverage': 1,
            'stoch_d_period': 3,
            'stoch_k_period': 14,
            'stoch_smooth_k': 3,
            'stop_atr_mult': 1.1,
            'supertrend_atr_period': 10,
            'supertrend_multiplier': 3.0,
            'tp_atr_mult': 2.7,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_k_period': ParameterSpec(
                name='stoch_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stoch_d_period': ParameterSpec(
                name='stoch_d_period',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stoch_smooth_k': ParameterSpec(
                name='stoch_smooth_k',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
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
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.7,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Indicator extraction
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])

        supertrend = indicators['supertrend']
        direction = np.nan_to_num(indicators['supertrend']["direction"])

        adx_val = np.nan_to_num(indicators['adx']["adx"])

        atr = np.nan_to_num(indicators['atr'])

        close = df["close"].values

        # Entry conditions
        long_mask = (k < 20) & (direction == 1) & (adx_val > 30)
        short_mask = (k > 80) & (direction == -1) & (adx_val > 30)

        # Exit conditions
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan  # k is float, NaN assignment safe
        cross_up = (k > 50) & (prev_k <= 50)
        cross_down = (k < 50) & (prev_k >= 50)
        stoch_cross = cross_up | cross_down

        prev_dir = np.roll(direction, 1)
        # avoid assigning NaN to integer array; copy first element instead
        prev_dir[0] = direction[0]
        dir_change = direction != prev_dir

        exit_mask = stoch_cross | dir_change

        # Apply exits first (override any entry)
        signals[exit_mask] = 0.0

        # Apply entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare ATR‑based stop‑loss / take‑profit columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 1.1)
        tp_mult = params.get("tp_atr_mult", 2.7)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals