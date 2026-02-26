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
         'atr_period': 14,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        adx_dict = indicators['adx']
        adx = np.nan_to_num(adx_dict["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Previous ROC for acceleration detection and zero‑cross detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan

        # Parameters (with defaults)
        adx_entry_thr = params.get("adx_entry_threshold", 25)
        adx_exit_thr = params.get("adx_exit_threshold", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.4)
        tp_atr_mult = params.get("tp_atr_mult", 2.9)

        # Entry conditions
        long_entry = (roc > 0) & (roc > prev_roc) & (adx > adx_entry_thr)
        short_entry = (roc < 0) & (roc < prev_roc) & (adx > adx_entry_thr)

        # Exit conditions: ROC crosses zero OR ADX falls below exit threshold
        cross_up = (roc > 0) & (prev_roc <= 0)
        cross_down = (roc < 0) & (prev_roc >= 0)
        exit_mask = (cross_up | cross_down) | (adx < adx_exit_thr)

        # Prevent duplicate consecutive signals
        prev_signal = np.roll(signals.values, 1)
        prev_signal[0] = 0.0
        long_entry &= (prev_signal != 1.0)
        short_entry &= (prev_signal != -1.0)

        # Apply exit mask (forces flat)
        signals[exit_mask] = 0.0

        # Apply entry masks
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Populate mask variables (optional external use)
        long_mask = long_entry
        short_mask = short_entry

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Write ATR‑based stop‑loss and take‑profit levels on entry bars
        long_entries = signals == 1.0
        short_entries = signals == -1.0

        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_atr_mult * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_atr_mult * atr[long_entries]

        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_atr_mult * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_atr_mult * atr[short_entries]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
