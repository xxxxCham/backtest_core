from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_pivot_atr_adx_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'pivot_points', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'atr_vol_threshold': 0.0015,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 6.0,
         'trailing_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0005,
                max_val=0.005,
                default=0.0015,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
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
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # price series
        close = df["close"].values

        # indicators (wrapped)
        atr = np.nan_to_num(indicators['atr'])

        pp = indicators['pivot_points']
        pivot = np.nan_to_num(pp["pivot"])
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])

        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # parameters
        atr_vol_thresh = float(params.get("atr_vol_threshold", 0.0015))
        adx_min = 25.0
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 6.0))

        # entry conditions
        long_mask = (close > r1) & (atr > atr_vol_thresh) & (adx_val > adx_min)
        short_mask = (close < s1) & (atr > atr_vol_thresh) & (adx_val > adx_min)

        # apply entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions: price crossing pivot or adx weakening
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_pivot = np.roll(pivot, 1)
        prev_pivot[0] = np.nan

        cross_up = (close > pivot) & (prev_close <= prev_pivot)
        cross_down = (close < pivot) & (prev_close >= prev_pivot)
        exit_mask = (cross_up | cross_down) | (adx_val < 20.0)

        # force flat on exit bars
        signals[exit_mask] = 0.0

        # initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # entry masks for numpy indexing
        entry_long_idx = (signals == 1.0).values
        entry_short_idx = (signals == -1.0).values

        # set ATR-based stop-loss and take-profit on entry bars
        df.loc[entry_long_idx, "bb_stop_long"] = close[entry_long_idx] - stop_atr_mult * atr[entry_long_idx]
        df.loc[entry_long_idx, "bb_tp_long"]   = close[entry_long_idx] + tp_atr_mult * atr[entry_long_idx]

        df.loc[entry_short_idx, "bb_stop_short"] = close[entry_short_idx] + stop_atr_mult * atr[entry_short_idx]
        df.loc[entry_short_idx, "bb_tp_short"]   = close[entry_short_idx] - tp_atr_mult * atr[entry_short_idx]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
