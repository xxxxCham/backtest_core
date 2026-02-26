from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_mfi_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_ma_period': 20,
         'atr_period': 14,
         'leverage': 1,
         'mfi_period': 14,
         'roc_period': 14,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.9,
         'warmup': 30}

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
            'atr_ma_period': ParameterSpec(
                name='atr_ma_period',
                min_val=10,
                max_val=50,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=30,
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
        # Initialize masks
        long_mask = np.zeros(len(df), dtype=bool)
        short_mask = np.zeros(len(df), dtype=bool)

        # Warmup protection (first 50 bars flat)
        signals.iloc[:50] = 0.0

        # Extract indicators with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # Previous ROC for acceleration detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan

        # ATR moving average (simple mean)
        atr_ma_period = int(params.get("atr_ma_period", 20))
        cumsum = np.cumsum(np.insert(atr, 0, 0.0))
        atr_ma = (cumsum[atr_ma_period:] - cumsum[:-atr_ma_period]) / atr_ma_period
        atr_ma_full = np.empty_like(atr)
        atr_ma_full[: atr_ma_period - 1] = np.nan
        atr_ma_full[atr_ma_period - 1 :] = atr_ma

        # Entry conditions
        long_cond = (roc > 0) & (roc > prev_roc) & (mfi > 50) & (atr > atr_ma_full)
        short_cond = (roc < 0) & (roc < prev_roc) & (mfi < 50) & (atr > atr_ma_full)

        long_mask[long_cond] = True
        short_mask[short_cond] = True

        # Exit condition: ROC crossing zero
        cross_up = (roc > 0) & (prev_roc <= 0)
        cross_down = (roc < 0) & (prev_roc >= 0)
        exit_mask = cross_up | cross_down

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR‑based stop‑loss and take‑profit levels
        stop_mult = float(params.get("stop_atr_mult", 1.4))
        tp_mult = float(params.get("tp_atr_mult", 2.9))
        close = df["close"].values

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
