from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mfi_adx_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['mfi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'mfi_period': 14,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                default=1.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
        # Extract required indicator arrays, ensuring NaNs are handled
        mfi = np.nan_to_num(indicators['mfi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Close price series as numpy array
        close = df["close"].values

        # Entry conditions
        long_mask = (mfi < 20) & (adx_val < 25)
        short_mask = (mfi > 80) & (adx_val < 25)

        # Exit condition (neutral MFI zone)
        exit_mask = (mfi >= 45) & (mfi <= 55)

        # Apply warm‑up protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Set signals: exit first to ensure it overrides any entry on the same bar
        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare ATR‑based stop‑loss / take‑profit columns (initialize with NaN)
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Retrieve risk‑management parameters with defaults
        stop_mult = float(params.get("stop_atr_mult", 1.2))
        tp_mult = float(params.get("tp_atr_mult", 2.5))

        # Compute and write SL/TP levels for long entries
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Compute and write SL/TP levels for short entries
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
