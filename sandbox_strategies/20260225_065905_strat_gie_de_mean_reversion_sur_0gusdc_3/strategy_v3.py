from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_mfi_atr_mean_reversion_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_ma_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'mfi_overbought': 70,
         'mfi_oversold': 30,
         'mfi_period': 14,
         'obv_lookback': 20,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.5,
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
            'mfi_oversold': ParameterSpec(
                name='mfi_oversold',
                min_val=10,
                max_val=40,
                default=30,
                param_type='int',
                step=1,
            ),
            'mfi_overbought': ParameterSpec(
                name='mfi_overbought',
                min_val=60,
                max_val=90,
                default=70,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract and clean indicators
        obv = np.nan_to_num(indicators['obv'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        obv_lookback = int(params.get("obv_lookback", 20))
        atr_ma_period = int(params.get("atr_ma_period", 14))
        mfi_oversold = float(params.get("mfi_oversold", 30))
        mfi_overbought = float(params.get("mfi_overbought", 70))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.2))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        # Rolling statistics for OBV
        obv_series = pd.Series(obv)
        obv_min = obv_series.rolling(obv_lookback, min_periods=obv_lookback).min().values
        obv_max = obv_series.rolling(obv_lookback, min_periods=obv_lookback).max().values
        obv_med = obv_series.rolling(obv_lookback, min_periods=obv_lookback).median().values

        # ATR moving average
        atr_series = pd.Series(atr)
        atr_ma = atr_series.rolling(atr_ma_period, min_periods=atr_ma_period).mean().values

        # Entry masks
        long_mask = (obv == obv_min) & (mfi < mfi_oversold) & (atr > atr_ma)
        short_mask = (obv == obv_max) & (mfi > mfi_overbought) & (atr > atr_ma)

        # Exit mask: OBV crossing its rolling median
        prev_obv = np.roll(obv, 1)
        prev_med = np.roll(obv_med, 1)
        prev_obv[0] = np.nan
        prev_med[0] = np.nan
        cross_up = (obv > obv_med) & (prev_obv <= prev_med)
        cross_down = (obv < obv_med) & (prev_obv >= prev_med)
        exit_mask = cross_up | cross_down

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0  # close any open position

        # ATR‑based SL/TP levels
        close = df["close"].values

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
