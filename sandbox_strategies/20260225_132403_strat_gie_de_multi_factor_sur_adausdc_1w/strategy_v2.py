from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_multi_factor_stoch_obv_adx_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'obv', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'stochastic_period': 14,
         'stochastic_smooth_d': 3,
         'stochastic_smooth_k': 3,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 5.1,
         'warmup': 50}
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
            'stochastic_smooth_k': ParameterSpec(
                name='stochastic_smooth_k',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_smooth_d': ParameterSpec(
                name='stochastic_smooth_d',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
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
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.1,
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

        # helper cross functions that support scalar thresholds
        def _ensure_array(x, y):
            if np.ndim(y) == 0:
                return np.full_like(x, y)
            return y

        def cross_up(x, y):
            y_arr = _ensure_array(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y_arr) & (prev_x <= prev_y)

        def cross_down(x, y):
            y_arr = _ensure_array(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y_arr) & (prev_x >= prev_y)

        # extract indicators
        stoch_k = np.nan_to_num(indicators['stochastic']["stoch_k"])
        obv = np.nan_to_num(indicators['obv'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # compute OBV SMA
        period = int(params.get("obv_sma_period", 14))
        if period < 1:
            period = 1
        obv_sma_full = np.convolve(
            obv, np.ones(period) / period, mode="valid"
        )
        obv_sma = np.empty(n, dtype=obv.dtype)
        obv_sma[: period - 1] = np.nan
        obv_sma[period - 1 :] = obv_sma_full

        # entry conditions
        long_mask = (stoch_k > 80) & (obv > obv_sma) & (adx > 25)
        short_mask = (stoch_k < 20) & (obv < obv_sma) & (adx > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        long_exit_mask = cross_down(stoch_k, 70) | cross_down(obv, obv_sma)
        short_exit_mask = cross_up(stoch_k, 30) | cross_up(obv, obv_sma)
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
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