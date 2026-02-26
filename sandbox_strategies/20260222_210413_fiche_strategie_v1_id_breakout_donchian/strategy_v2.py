from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_with_roc')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is required for stop‑loss and take‑profit calculations
        return ['donchian', 'adx', 'roc', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 2.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Retrieve indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        donchian = indicators['donchian']
        upper = np.nan_to_num(indicators['donchian']["upper"])
        lower = np.nan_to_num(indicators['donchian']["lower"])
        middle = np.nan_to_num(indicators['donchian']["middle"])

        adx_val = np.nan_to_num(indicators['adx']["adx"])
        roc_arr = np.nan_to_num(indicators['roc'])

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        # Long and short entry masks
        long_mask = (close > upper) & (adx_val > 30) & (roc_arr > 0)
        short_mask = (close < lower) & (adx_val > 30) & (roc_arr < 0)

        # Apply entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic: cross of close over/under middle or weak ADX
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        cross_up = (close > middle) & (prev_close <= prev_middle)
        cross_down = (close < middle) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down

        exit_mask = cross_any | (adx_val < 20)
        signals[exit_mask] = 0.0

        # Ensure no signals during warmup
        signals.iloc[:warmup] = 0.0

        # ATR‑based stop‑loss and take‑profit levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals