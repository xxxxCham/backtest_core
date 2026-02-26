from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_mfi_adx_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'mfi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_exit': 20,
         'adx_min': 25,
         'adx_period': 14,
         'leverage': 1,
         'mfi_over': 55,
         'mfi_period': 14,
         'mfi_under': 45,
         'roc_period': 12,
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
                default=12,
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
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_min': ParameterSpec(
                name='adx_min',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_exit': ParameterSpec(
                name='adx_exit',
                min_val=10,
                max_val=30,
                default=20,
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

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicators with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        mfi = np.nan_to_num(indicators['mfi'])
        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Previous ROC for acceleration detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan

        # Parameters
        adx_min = float(params.get("adx_min", 25))
        adx_exit = float(params.get("adx_exit", 20))
        mfi_over = float(params.get("mfi_over", 55))
        mfi_under = float(params.get("mfi_under", 45))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.4))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.9))

        # Entry conditions
        long_mask = (roc > 0) & (roc > prev_roc) & (mfi > mfi_over) & (adx_vals > adx_min)
        short_mask = (roc < 0) & (roc < prev_roc) & (mfi < mfi_under) & (adx_vals > adx_min)

        # Exit conditions: ROC crossing zero or ADX dropping below exit threshold
        cross_up_zero = (roc > 0) & (prev_roc <= 0)
        cross_down_zero = (roc < 0) & (prev_roc >= 0)
        roc_cross_zero = cross_up_zero | cross_down_zero
        exit_mask = roc_cross_zero | (adx_vals < adx_exit)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Initialize SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        close = df["close"].values

        # Write ATR‑based stop‑loss and take‑profit on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
