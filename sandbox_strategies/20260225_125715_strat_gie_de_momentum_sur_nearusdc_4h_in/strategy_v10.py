from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_adx_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'leverage': 1,
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
        # extract and sanitize indicators
        roc = np.nan_to_num(indicators['roc'])
        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # previous ROC for acceleration and exit detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan

        # entry conditions
        long_cond = (roc > 0) & (roc > prev_roc) & (adx_vals > 25)
        short_cond = (roc < 0) & (roc < prev_roc) & (adx_vals > 25)

        # suppress consecutive identical entries
        prev_long = np.roll(long_cond, 1)
        prev_long[0] = False
        prev_short = np.roll(short_cond, 1)
        prev_short[0] = False
        long_cond = long_cond & (~prev_long)
        short_cond = short_cond & (~prev_short)

        # exit condition (momentum reversal or weakening trend)
        exit_cond = (roc * prev_roc <= 0) | (adx_vals < 20)

        # assign masks to pre‑allocated arrays
        long_mask[:] = long_cond
        short_mask[:] = short_cond

        # generate signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # exit overrides any entry on the same bar
        signals[exit_cond] = 0.0

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop‑loss / take‑profit levels
        stop_mult = float(params.get("stop_atr_mult", 1.4))
        tp_mult = float(params.get("tp_atr_mult", 2.9))

        # initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # write levels only on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
