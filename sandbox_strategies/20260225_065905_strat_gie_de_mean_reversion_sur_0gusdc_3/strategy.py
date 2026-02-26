from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_mfi_atr_mean_reversion_v4')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_ma_period': 10,
         'atr_period': 14,
         'leverage': 1,
         'mfi_overbought': 80,
         'mfi_oversold': 20,
         'mfi_period': 12,
         'obv_lookback': 20,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.6,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'obv_lookback': ParameterSpec(
                name='obv_lookback',
                min_val=10,
                max_val=50,
                default=25,
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
                default=25,
                param_type='int',
                step=1,
            ),
            'mfi_overbought': ParameterSpec(
                name='mfi_overbought',
                min_val=60,
                max_val=90,
                default=75,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
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
        # Initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract and sanitise indicators
        obv = np.nan_to_num(indicators['obv'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        obv_lookback = int(params.get("obv_lookback", 25))
        atr_ma_period = int(params.get("atr_ma_period", 14))
        mfi_overbought = params.get("mfi_overbought", 75)
        mfi_oversold = params.get("mfi_oversold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 2.4)

        # Rolling OBV extremes and median
        obv_series = pd.Series(obv)
        obv_min = np.nan_to_num(obv_series.rolling(obv_lookback, min_periods=1).min().values)
        obv_max = np.nan_to_num(obv_series.rolling(obv_lookback, min_periods=1).max().values)
        obv_median = np.nan_to_num(obv_series.rolling(obv_lookback, min_periods=1).median().values)

        # ATR moving average
        atr_series = pd.Series(atr)
        atr_ma = np.nan_to_num(atr_series.rolling(atr_ma_period, min_periods=1).mean().values)

        # Entry conditions
        long_cond = (obv == obv_min) & (mfi < mfi_oversold) & (atr > atr_ma)
        short_cond = (obv == obv_max) & (mfi > mfi_overbought) & (atr > atr_ma)

        long_mask[long_cond] = True
        short_mask[short_cond] = True

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit condition: OBV crossing its rolling median
        prev_obv = np.roll(obv, 1)
        prev_median = np.roll(obv_median, 1)
        prev_obv[0] = np.nan
        prev_median[0] = np.nan

        cross_up = (obv > obv_median) & (prev_obv <= prev_median)
        cross_down = (obv < obv_median) & (prev_obv >= prev_median)
        exit_mask = cross_up | cross_down

        signals[exit_mask] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values

        # Write ATR‑based stop‑loss and take‑profit levels on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
