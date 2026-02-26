from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'bollinger_period': 20,
            'bollinger_std': 2,
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
            'warmup': 50,
        }

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
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std': ParameterSpec(
                name='bollinger_std',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
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
                max_val=6.0,
                default=3.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get('warmup', 50))

        # Pull indicator arrays
        close = df["close"].values
        # Ensure direction is float to allow NaN assignment
        direction = np.array(indicators['supertrend']["direction"], dtype=float)
        upper = np.array(indicators['bollinger']["upper"], dtype=float)
        middle = np.array(indicators['bollinger']["middle"], dtype=float)
        lower = np.array(indicators['bollinger']["lower"], dtype=float)
        rsi = np.array(indicators['rsi'], dtype=float)
        atr = np.array(indicators['atr'], dtype=float)

        # Entry conditions
        long_mask = (direction == 1.0) & (close > upper) & (rsi > 50.0)
        short_mask = (direction == -1.0) & (close < lower) & (rsi < 50.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down_middle = (close < middle) & (prev_close >= middle)

        prev_dir = np.roll(direction, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change = direction != prev_dir

        exit_mask = cross_down_middle | dir_change
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels on entry bars
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        # Initialize columns if they don't exist
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]

        signals.iloc[:warmup] = 0.0
        return signals