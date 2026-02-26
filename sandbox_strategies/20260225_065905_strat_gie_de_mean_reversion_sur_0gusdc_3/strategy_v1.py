from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_mfi_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'mfi_overbought': 70,
         'mfi_oversold': 30,
         'mfi_period': 14,
         'obv_lookback': 20,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.4,
         'warmup': 30}

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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # extract indicators safely
        obv = np.nan_to_num(indicators['obv'])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # rolling high/low of OBV over lookback window
        lookback = int(params.get("obv_lookback", 20))
        obv_high = pd.Series(obv).rolling(window=lookback, min_periods=lookback).max().values
        obv_low = pd.Series(obv).rolling(window=lookback, min_periods=lookback).min().values

        # parameter thresholds
        mfi_oversold = params.get("mfi_oversold", 30)
        mfi_overbought = params.get("mfi_overbought", 70)

        # entry conditions
        long_mask = (obv == obv_high) & (mfi < mfi_oversold)
        short_mask = (obv == obv_low) & (mfi > mfi_overbought)

        # exit condition (neutral OBV zone)
        exit_mask = (obv < obv_high) & (obv > obv_low)

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR‑based stop‑loss / take‑profit levels
        stop_mult = params.get("stop_atr_mult", 1.1)
        tp_mult = params.get("tp_atr_mult", 2.4)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
