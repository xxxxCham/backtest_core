from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='donchian_adx_atr_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'donchian_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.86,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=20,
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
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.86,
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
        # Extract indicator arrays with nan handling
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr_arr = np.nan_to_num(indicators['atr'])
        donch = indicators['donchian']
        upper = np.nan_to_num(donch["upper"])
        middle = np.nan_to_num(donch["middle"])
        lower = np.nan_to_num(donch["lower"])
        close_arr = df["close"].values

        # Helper for cross detection
        prev_close = np.roll(close_arr, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan

        # Entry conditions
        cross_up = (close_arr > upper) & (prev_close <= prev_upper)
        cross_down = (close_arr < lower) & (prev_close >= prev_lower)

        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        long_mask = cross_up & (adx_val > 25)
        short_mask = cross_down & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions are handled by stop‑loss/take‑profit columns

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute ATR-based SL/TP on entry bars
        stop_atr_mult = float(params.get("stop_atr_mult", 1.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.86))

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close_arr[long_entry] - stop_atr_mult * atr_arr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close_arr[long_entry] + tp_atr_mult * atr_arr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close_arr[short_entry] + stop_atr_mult * atr_arr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close_arr[short_entry] - tp_atr_mult * atr_arr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
