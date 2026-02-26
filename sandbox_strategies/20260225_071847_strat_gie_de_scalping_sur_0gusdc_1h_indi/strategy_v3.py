from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_vwap_atr_scalp_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 20,
            'leverage': 1,
            'stop_atr_mult': 1.8,
            'tp_atr_mult': 5.04,
            'vwap_window': 20,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vwap_window': ParameterSpec(
                name='vwap_window',
                min_val=10,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.04,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # --- preparation ---
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        vwap = np.nan_to_num(indicators['vwap'])
        atr = np.nan_to_num(indicators['atr'])

        # previous values for cross detection
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        prev_close[0] = np.nan
        prev_ema[0] = np.nan

        cross_up = (close > ema) & (prev_close <= prev_ema)
        cross_down = (close < ema) & (prev_close >= prev_ema)

        # entry conditions
        long_entry = cross_up & (close > vwap)
        short_entry = cross_down & (close < vwap)

        # masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warm‑up protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # risk‑management parameters
        stop_atr_mult = float(params.get("stop_atr_mult", 1.8))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.04))

        # initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute SL/TP on entry bars using positional indexing
        if long_mask.any():
            idx_long = np.where(long_mask)[0]
            df.iloc[idx_long, df.columns.get_loc("bb_stop_long")] = (
                close[idx_long] - stop_atr_mult * atr[idx_long]
            )
            df.iloc[idx_long, df.columns.get_loc("bb_tp_long")] = (
                close[idx_long] + tp_atr_mult * atr[idx_long]
            )

        if short_mask.any():
            idx_short = np.where(short_mask)[0]
            df.iloc[idx_short, df.columns.get_loc("bb_stop_short")] = (
                close[idx_short] + stop_atr_mult * atr[idx_short]
            )
            df.iloc[idx_short, df.columns.get_loc("bb_tp_short")] = (
                close[idx_short] - tp_atr_mult * atr[idx_short]
            )

        return signals