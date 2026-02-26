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
        return {'adx_period': 17,
         'atr_period': 14,
         'donchian_period': 35,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 6.0,
         'warmup': 35}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=35,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=17,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=10,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=6.0,
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
        # Prepare indicator arrays
        close = np.nan_to_num(df["close"].values)
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        lower = np.nan_to_num(dc["lower"])
        middle = np.nan_to_num(dc["middle"])
        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Cross detection helper
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        cross_up = (close > middle) & (prev_close <= prev_middle)
        cross_down = (close < middle) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down

        # Long and short entry masks
        long_mask = (close > upper) & (adx_val > 20)
        short_mask = (close < lower) & (adx_val > 20)

        # Exit mask
        exit_mask = cross_any | (adx_val < 15)

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

        # Compute ATR-based SL/TP on entry bars
        entry_mask = long_mask | short_mask
        if entry_mask.any():
            stop_mult = params.get("stop_atr_mult", 1.5)
            tp_mult = params.get("tp_atr_mult", 6.0)

            # Long positions
            long_entry = long_mask[entry_mask]
            df.loc[entry_mask, "bb_stop_long"] = np.where(
                long_entry,
                close[entry_mask] - stop_mult * atr[entry_mask],
                np.nan,
            )
            df.loc[entry_mask, "bb_tp_long"] = np.where(
                long_entry,
                close[entry_mask] + tp_mult * atr[entry_mask],
                np.nan,
            )

            # Short positions
            short_entry = short_mask[entry_mask]
            df.loc[entry_mask, "bb_stop_short"] = np.where(
                short_entry,
                close[entry_mask] + stop_mult * atr[entry_mask],
                np.nan,
            )
            df.loc[entry_mask, "bb_tp_short"] = np.where(
                short_entry,
                close[entry_mask] - tp_mult * atr[entry_mask],
                np.nan,
            )
        signals.iloc[:warmup] = 0.0
        return signals
