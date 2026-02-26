from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_stochastic_roc_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'stochastic', 'roc', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'momentum_period': 14,
            'roc_period': 14,
            'stochastic_d_period': 3,
            'stochastic_k_period': 14,
            'stop_atr_mult': 1.9,
            'tp_atr_mult': 2.4,
            'warmup': 30,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.9,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=4.0,
                default=2.4,
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
        signals.iloc[:warmup] = 0.0

        # Helper cross functions that guard against scalar inputs
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 0 or y.ndim == 0:
                return np.full(x.size if x.ndim else 1, False, dtype=bool)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 0 or y.ndim == 0:
                return np.full(x.size if x.ndim else 1, False, dtype=bool)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Extract indicator arrays
        momentum = np.nan_to_num(indicators['momentum'])
        stoch_k = np.nan_to_num(indicators['stochastic']['stoch_k'])
        roc = np.nan_to_num(indicators['roc'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (momentum > 0) & (stoch_k < 30) & (roc > 0)
        short_mask = (momentum < 0) & (stoch_k > 70) & (roc < 0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        long_exit_mask = (
            cross_down(momentum, 0)
            | cross_up(stoch_k, 70)
            | (roc < 0)
        )
        short_exit_mask = (
            cross_up(momentum, 0)
            | cross_down(stoch_k, 30)
            | (roc > 0)
        )
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.9))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.4))

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = (
            close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = (
            close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        )
        df.loc[entry_short_mask, "bb_stop_short"] = (
            close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = (
            close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        )

        signals.iloc[:warmup] = 0.0
        return signals