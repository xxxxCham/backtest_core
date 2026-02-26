from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='pivot_breakout_atr_filtered')

    @property
    def required_indicators(self) -> List[str]:
        return ['pivot_points', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_exit_threshold': 20,
         'adx_period': 14,
         'adx_threshold': 25,
         'atr_period': 14,
         'atr_vol_threshold': 0.0005,
         'leverage': 1,
         'stop_atr_mult': 2.3,
         'tp_atr_mult': 5.8,
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
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.1,
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
        # Extract needed indicator arrays
        close = df["close"].values
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        adx_thr = params.get("adx_threshold", 25)
        adx_exit_thr = params.get("adx_exit_threshold", 20)
        atr_vol_thr = params.get("atr_vol_threshold", 0.0005)
        stop_mult = params.get("stop_atr_mult", 2.3)
        tp_mult = params.get("tp_atr_mult", 5.8)

        # Entry masks
        long_mask[:] = (close > r1) & (adx_arr >= adx_thr) & (atr > atr_vol_thr)
        short_mask[:] = (close < s1) & (adx_arr >= adx_thr) & (atr > atr_vol_thr)

        # Apply entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit masks (price re‑enters range or ADX weakens)
        exit_long = ((close <= r1) & (close >= s1)) | (adx_arr < adx_exit_thr)
        exit_short = ((close >= s1) & (close <= r1)) | (adx_arr < adx_exit_thr)

        # Apply exit signals (flatten position)
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR‑based stop‑loss and take‑profit on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
