from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_mfi_mean_reversion_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_ma_period': 14,
         'leverage': 1,
         'mfi_overbought': 70,
         'mfi_oversold': 30,
         'mfi_period': 14,
         'obv_lookback': 20,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.4,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_lookback': ParameterSpec(
                name='obv_lookback',
                min_val=10,
                max_val=50,
                default=20,
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
            'atr_ma_period': ParameterSpec(
                name='atr_ma_period',
                min_val=7,
                max_val=21,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.4,
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
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays with NaN handling
        obv = np.nan_to_num(indicators['obv'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        obv_lookback = int(params.get("obv_lookback", 20))
        atr_ma_period = int(params.get("atr_ma_period", 14))
        mfi_oversold = float(params.get("mfi_oversold", 30))
        mfi_overbought = float(params.get("mfi_overbought", 70))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.1))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.4))

        # Compute rolling low/high for OBV using sliding window view
        if obv_lookback > 1 and n >= obv_lookback:
            obv_windows = np.lib.stride_tricks.sliding_window_view(obv, obv_lookback)
            obv_low_roll = np.min(obv_windows, axis=1)
            obv_high_roll = np.max(obv_windows, axis=1)
            pad = np.full(obv_lookback - 1, np.nan)
            obv_low = np.concatenate([pad, obv_low_roll])
            obv_high = np.concatenate([pad, obv_high_roll])
        else:
            obv_low = np.full(n, np.nan)
            obv_high = np.full(n, np.nan)

        # Compute ATR moving average
        if atr_ma_period > 1 and n >= atr_ma_period:
            atr_windows = np.lib.stride_tricks.sliding_window_view(atr, atr_ma_period)
            atr_ma = np.concatenate([np.full(atr_ma_period - 1, np.nan), np.mean(atr_windows, axis=1)])
        else:
            atr_ma = np.full(n, np.nan)

        atr_condition = atr > atr_ma

        # Long entry: OBV at recent low, MFI oversold, ATR condition
        long_mask = (obv == obv_low) & (mfi < mfi_oversold) & atr_condition

        # Short entry: OBV at recent high, MFI overbought, ATR condition
        short_mask = (obv == obv_high) & (mfi > mfi_overbought) & atr_condition

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare SL/TP columns (initialize with NaN)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values

        # Write ATR‑based stop‑loss and take‑profit for long entries
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Write ATR‑based stop‑loss and take‑profit for short entries
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
