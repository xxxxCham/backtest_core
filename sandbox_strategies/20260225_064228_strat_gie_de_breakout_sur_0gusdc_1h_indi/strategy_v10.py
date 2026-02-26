from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='pivot_breakout_volume_adx_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['pivot_points', 'adx', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_exit_threshold': 20,
         'adx_period': 14,
         'adx_threshold': 25,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 2.3,
         'tp_atr_mult': 5.8,
         'vol_osc_long_period': 28,
         'vol_osc_short_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_exit_threshold': ParameterSpec(
                name='adx_exit_threshold',
                min_val=5,
                max_val=30,
                default=20,
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
            'vol_osc_short_period': ParameterSpec(
                name='vol_osc_short_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'vol_osc_long_period': ParameterSpec(
                name='vol_osc_long_period',
                min_val=10,
                max_val=60,
                default=28,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=5.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.8,
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
        # extract needed series/arrays
        close = df["close"].values
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])
        vol = np.nan_to_num(indicators['volume_oscillator'])
        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # parameters with defaults
        adx_thr = float(params.get("adx_threshold", 25))
        adx_exit_thr = float(params.get("adx_exit_threshold", 20))
        stop_mult = float(params.get("stop_atr_mult", 2.3))
        tp_mult = float(params.get("tp_atr_mult", 5.8))

        # entry conditions
        long_cond = (close > r1) & (vol > 0) & (adx_vals > adx_thr)
        short_cond = (close < s1) & (vol < 0) & (adx_vals > adx_thr)

        # apply to masks
        long_mask[long_cond] = True
        short_mask[short_cond] = True

        # set signal values
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # write ATR‑based SL/TP only on entry bars
        # ensure columns exist
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        # exit condition (flat) – signals already 0 elsewhere, but ensure we clear any lingering positions
        exit_cond = ((close <= r1) & (close >= s1)) | (adx_vals < adx_exit_thr)
        # when exit condition true, ensure no signal (overwrites any accidental entry)
        signals[exit_cond] = 0.0
        long_mask[exit_cond] = False
        short_mask[exit_cond] = False

        # warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
