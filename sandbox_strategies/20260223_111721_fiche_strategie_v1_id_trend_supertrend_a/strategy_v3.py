from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_ema_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 20,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.25,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # wrap indicator arrays
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        ema_val = np.nan_to_num(indicators['ema'])
        atr_val = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # entry conditions
        long_mask = (st_dir == 1) & (adx_val > 30) & (close > ema_val)
        short_mask = (st_dir == -1) & (adx_val > 30) & (close < ema_val)

        # helper for cross detection
        prev_close = np.roll(close, 1); prev_close[0] = np.nan
        prev_ema = np.roll(ema_val, 1); prev_ema[0] = np.nan
        cross_up = (close > ema_val) & (prev_close <= prev_ema)
        cross_down = (close < ema_val) & (prev_close >= prev_ema)
        cross_any = cross_up | cross_down

        # exit conditions
        prev_dir = np.roll(st_dir, 1); prev_dir[0] = 0.0
        dir_change = (st_dir != prev_dir) & (prev_dir != 0.0)
        exit_mask = dir_change | (adx_val < 20) | cross_any

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        stop_atr_mult = float(params.get("stop_atr_mult", 1.25))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        entry_mask = long_mask | short_mask

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr_val[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr_val[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr_val[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr_val[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
