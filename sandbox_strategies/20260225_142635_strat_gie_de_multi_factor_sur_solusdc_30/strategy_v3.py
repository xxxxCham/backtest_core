from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_stochastic_atr_multi_factor')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_threshold': 0.0005,
            'leverage': 1,
            'stochastic_d_period': 3,
            'stochastic_k_period': 3,
            'stochastic_overbought': 80,
            'stochastic_oversold': 20,
            'stochastic_period': 14,
            'stop_atr_mult': 1.6,
            'tp_atr_mult': 2.9,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_period': ParameterSpec(
                name='stochastic_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_oversold': ParameterSpec(
                name='stochastic_oversold',
                min_val=5,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'stochastic_overbought': ParameterSpec(
                name='stochastic_overbought',
                min_val=70,
                max_val=95,
                default=80,
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
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
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
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.9,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
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

        # Helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            prev_y = np.roll(y, 1)
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            prev_y = np.roll(y, 1)
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Extract indicators
        obv = np.nan_to_num(indicators['obv'])
        stoch_k = np.nan_to_num(indicators['stochastic']["stoch_k"])
        atr = np.nan_to_num(indicators['atr'])

        # OBV trend masks
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan
        obv_rising = obv > prev_obv
        obv_falling = obv < prev_obv

        # Entry conditions
        long_mask = (
            obv_rising
            & cross_up(stoch_k, np.full(n, params["stochastic_oversold"], dtype=float))
            & (atr > params["atr_threshold"])
        )
        short_mask = (
            obv_falling
            & cross_down(stoch_k, np.full(n, params["stochastic_overbought"], dtype=float))
            & (atr > params["atr_threshold"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        long_exit = cross_down(stoch_k, np.full(n, 50.0, dtype=float)) | obv_falling
        short_exit = cross_up(stoch_k, np.full(n, 50.0, dtype=float)) | obv_rising
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals