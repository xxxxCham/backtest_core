from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_mfi_atr_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'mfi_period': 14,
         'roc_period': 14,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'mfi_period': ParameterSpec(
                name='mfi_period',
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
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.9,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=30,
                max_val=120,
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
        # extract indicator arrays
        roc = np.nan_to_num(indicators['roc'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # previous values for acceleration and cross detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan
        prev_mfi = np.roll(mfi, 1)
        prev_mfi[0] = np.nan

        # entry conditions
        long_entry = (roc > 0) & (roc > prev_roc) & (mfi > 55) & (atr > 0)
        short_entry = (roc < 0) & (roc < prev_roc) & (mfi < 45) & (atr > 0)

        # apply warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            long_entry[:warmup] = False
            short_entry[:warmup] = False

        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # set signals for entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions: ROC or MFI crossing their mid levels
        cross_zero_up = (roc > 0) & (prev_roc <= 0)
        cross_zero_down = (roc < 0) & (prev_roc >= 0)
        cross_zero = cross_zero_up | cross_zero_down

        cross_mfi_up = (mfi > 50) & (prev_mfi <= 50)
        cross_mfi_down = (mfi < 50) & (prev_mfi >= 50)
        cross_mfi = cross_mfi_up | cross_mfi_down

        exit_mask = cross_zero | cross_mfi
        signals[exit_mask] = 0.0

        # ATR‑based stop‑loss / take‑profit levels
        stop_atr_mult = float(params.get("stop_atr_mult", 1.4))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.9))

        # initialize columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # write levels only on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
