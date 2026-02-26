from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_roc_mfi_atr_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'roc', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'mfi_period': 14,
         'roc_period': 14,
         'stochastic_period': 14,
         'stochastic_smooth_d': 3,
         'stochastic_smooth_k': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.7,
         'warmup': 20}

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
                default=1.5,
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
        # Boolean masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        roc_arr = np.nan_to_num(indicators['roc'])
        mfi_arr = np.nan_to_num(indicators['mfi'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        # Entry conditions
        long_mask = (k > d) & (k > 80) & (roc_arr > 0.5) & (mfi_arr > 70)
        short_mask = (k < d) & (k < 20) & (roc_arr < -0.5) & (mfi_arr < 30)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Skip warmup to avoid NaNs
        signals.iloc[:50] = 0.0

        # Initialize ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Parameters for SL/TP
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.7))

        # Set SL/TP levels only on entry bars
        df.loc[long_mask, "bb_stop_long"] = close_arr[long_mask] - stop_atr_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close_arr[long_mask] + tp_atr_mult * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close_arr[short_mask] + stop_atr_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close_arr[short_mask] - tp_atr_mult * atr_arr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
