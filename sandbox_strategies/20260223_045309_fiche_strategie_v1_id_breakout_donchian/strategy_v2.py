from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold_entry': 30,
         'adx_threshold_exit': 25,
         'donchian_period': 25,
         'leverage': 1,
         'stop_atr_mult': 1.75,
         'tp_atr_mult': 5.5,
         'warmup': 25}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_threshold_entry': ParameterSpec(
                name='adx_threshold_entry',
                min_val=20,
                max_val=40,
                default=30,
                param_type='int',
                step=1,
            ),
            'adx_threshold_exit': ParameterSpec(
                name='adx_threshold_exit',
                min_val=10,
                max_val=30,
                default=25,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.5,
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
        # Initialize boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        close = df["close"].values
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        lower = np.nan_to_num(dc["lower"])
        middle = np.nan_to_num(dc["middle"])

        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr_arr = np.nan_to_num(indicators['atr'])

        # Parameters
        adx_entry = params.get("adx_threshold_entry", 30)
        adx_exit = params.get("adx_threshold_exit", 25)
        stop_atr_mult = params.get("stop_atr_mult", 1.75)
        tp_atr_mult = params.get("tp_atr_mult", 5.5)

        # Entry conditions
        long_mask = (close > upper) & (adx_arr > adx_entry)
        short_mask = (close < lower) & (adx_arr > adx_entry)

        # Cross any helper
        prev_close = np.roll(close, 1)
        prev_mid = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_mid[0] = np.nan
        cross_up = (close > middle) & (prev_close <= prev_mid)
        cross_down = (close < middle) & (prev_close >= prev_mid)
        cross_any = cross_up | cross_down

        # Exit condition
        exit_mask = cross_any | (adx_arr < adx_exit)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        entry_long_mask = long_mask
        entry_short_mask = short_mask
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr_arr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr_arr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr_arr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr_arr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
