from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rsi_obv_atr_volatility_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
                default=3.9,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # wrap indicator arrays
        rsi = np.nan_to_num(indicators['rsi'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])

        # rolling mean of ATR using convolution (window 14)
        atr_mean = np.convolve(atr, np.ones(14) / 14, mode="same")

        # previous OBV for direction check
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # long entry: rsi<30 AND obv>prev_obv AND atr>atr_mean
        long_mask = (rsi < 30.0) & (obv > prev_obv) & (atr > atr_mean)

        # short entry: rsi>70 AND obv<prev_obv AND atr>atr_mean
        short_mask = (rsi > 70.0) & (obv < prev_obv) & (atr > atr_mean)

        # exit conditions
        # cross of rsi with 50
        rsi_cross_50 = np.full(n, 50.0)
        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan
        cross_up = (rsi > rsi_cross_50) & (prev_rsi <= rsi_cross_50)
        cross_down = (rsi < rsi_cross_50) & (prev_rsi >= rsi_cross_50)
        rsi_exit = cross_up | cross_down

        # obv direction reversal
        obv_diff = obv - prev_obv
        obv_diff[0] = np.nan
        prev_diff = np.roll(obv_diff, 1)
        prev_diff[0] = np.nan
        obv_rev = ((obv_diff > 0) & (prev_diff < 0)) | ((obv_diff < 0) & (prev_diff > 0))

        exit_mask = rsi_exit | obv_rev

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # warmup
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.9))
        close = df["close"].values

        # initialize columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
