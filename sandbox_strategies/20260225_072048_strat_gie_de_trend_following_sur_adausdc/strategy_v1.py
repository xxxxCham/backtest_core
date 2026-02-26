from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_sma_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.7,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
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
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.7,
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
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # extract indicators
        vx = indicators['vortex']
        vip = np.nan_to_num(vx["vi_plus"])
        vim = np.nan_to_num(vx["vi_minus"])
        sma_arr = np.nan_to_num(indicators['sma'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # entry masks
        long_mask = (vip > vim) & (close > sma_arr)
        short_mask = (vim > vip) & (close < sma_arr)

        # vortex cross detection for exit
        prev_vip = np.roll(vip, 1)
        prev_vim = np.roll(vim, 1)
        prev_vip[0] = np.nan
        prev_vim[0] = np.nan
        cross_up = (vip > vim) & (prev_vip <= prev_vim)
        cross_down = (vip < vim) & (prev_vip >= prev_vim)
        exit_mask = cross_up | cross_down

        # apply signals
        signals[exit_mask] = 0.0
        signals[long_mask & ~exit_mask] = 1.0
        signals[short_mask & ~exit_mask] = -1.0

        # ATR‑based stop‑loss / take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.0))
        tp_mult = float(params.get("tp_atr_mult", 2.7))

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr_arr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr_arr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr_arr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr_arr[entry_short]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
