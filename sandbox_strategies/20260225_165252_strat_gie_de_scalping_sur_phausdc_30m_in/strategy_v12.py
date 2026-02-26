from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='scalping_ema_stoch_atr_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'ema_period': 20,
            'leverage': 1,
            'stoch_d_period': 3,
            'stoch_k_period': 5,
            'stoch_smooth_k': 3,
            'stop_atr_mult': 1.0,
            'tp_atr_mult': 1.8,
            'warmup': 30
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
            'stoch_k_period': ParameterSpec(
                name='stoch_k_period',
                min_val=5,
                max_val=20,
                default=5,
                param_type='int',
                step=1,
            ),
            'stoch_d_period': ParameterSpec(
                name='stoch_d_period',
                min_val=3,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stoch_smooth_k': ParameterSpec(
                name='stoch_smooth_k',
                min_val=1,
                max_val=5,
                default=3,
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
                max_val=3.0,
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=60,
                default=30,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Indicator arrays
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])

        # Helper cross functions
        def cross_up(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # EMA slope
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        ema_slope_pos = ema > prev_ema
        ema_slope_neg = ema < prev_ema

        # Entry conditions
        long_mask = (
            cross_up(close, ema)
            & ema_slope_pos
            & (k < 20)
            & cross_up(k, d)
        )
        short_mask = (
            cross_down(close, ema)
            & ema_slope_neg
            & (k > 80)
            & cross_down(k, d)
        )

        # Exit conditions
        # Helper to detect crossing of a scalar value
        def cross_scalar_up(x, scalar):
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            return (x > scalar) & (prev_x <= scalar)

        def cross_scalar_down(x, scalar):
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            return (x < scalar) & (prev_x >= scalar)

        exit_long_mask = (
            cross_down(close, ema)
            | cross_scalar_up(k, 50)
            | cross_scalar_down(k, 50)
        )
        exit_short_mask = (
            cross_up(close, ema)
            | cross_scalar_up(k, 50)
            | cross_scalar_down(k, 50)
        )

        # Combine signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask & (signals == 1.0)] = 0.0
        signals[exit_short_mask & (signals == -1.0)] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]

        signals.iloc[:warmup] = 0.0
        return signals