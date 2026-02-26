from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_mfi_macd_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'mfi', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
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
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=20,
                default=9,
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
        # Initialize signals series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Prepare masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup handling
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0
            long_mask[:warmup] = False
            short_mask[:warmup] = False

        # Extract indicator arrays with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        mfi = np.nan_to_num(indicators['mfi'])
        macd_dict = indicators['macd']
        macd_hist = np.nan_to_num(macd_dict["histogram"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Previous values for acceleration and cross detection
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan

        # Entry conditions
        long_entry = (roc > 0) & (roc > prev_roc) & (mfi > 50) & (macd_hist > 0)
        short_entry = (roc < 0) & (roc < prev_roc) & (mfi < 50) & (macd_hist < 0)

        # Apply entry masks
        long_mask = long_entry
        short_mask = short_entry

        # Exit conditions (crosses zero or MACD histogram sign change)
        roc_cross_up = (roc > 0) & (prev_roc <= 0)
        roc_cross_down = (roc < 0) & (prev_roc >= 0)
        roc_cross = roc_cross_up | roc_cross_down

        macd_cross_up = (macd_hist > 0) & (prev_hist <= 0)
        macd_cross_down = (macd_hist < 0) & (prev_hist >= 0)
        macd_cross = macd_cross_up | macd_cross_down

        exit_mask = roc_cross | macd_cross

        # Ensure exits reset any existing signals (signals already 0.0 by default)
        signals[exit_mask] = 0.0

        # Set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop‑loss and take‑profit levels
        stop_mult = float(params.get("stop_atr_mult", 1.4))
        tp_mult = float(params.get("tp_atr_mult", 2.9))

        # Long SL/TP
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
