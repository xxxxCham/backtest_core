from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_bollinger')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'bollinger', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.0, 'tp_atr_mult': 5.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=1.0,
                max_val=10.0,
                default=5.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=10,
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

        # --- indicator arrays -------------------------------------------------
        close = df["close"].values
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        atr = np.nan_to_num(indicators['atr'])

        # --- entry masks ------------------------------------------------------
        long_mask = (close > upper) & (st_dir == 1) & (adx_val > 30)
        short_mask = (close < lower) & (st_dir == -1) & (adx_val > 30)

        # --- exit masks -------------------------------------------------------
        # shift direction and allow NaN for first element
        prev_dir = np.roll(st_dir.astype(float), 1)
        prev_dir[0] = np.nan
        dir_change = (~np.isnan(prev_dir)) & (st_dir != prev_dir)

        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        cross_middle_up = (close > middle) & (prev_close <= prev_middle)
        cross_middle_down = (close < middle) & (prev_close >= prev_middle)
        cross_middle = cross_middle_up | cross_middle_down

        adx_exit = adx_val < 25
        exit_mask = dir_change | cross_middle | adx_exit

        # --- assign signals ---------------------------------------------------
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # --- warmup period ----------------------------------------------------
        signals.iloc[:warmup] = 0.0

        # --- prepare SL/TP columns --------------------------------------------
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # --- ATR-based SL/TP for long entries --------------------------------
        long_entry = signals == 1.0
        if long_entry.any():
            df.loc[long_entry, "bb_stop_long"] = close[long_entry] - params["stop_atr_mult"] * atr[long_entry]
            df.loc[long_entry, "bb_tp_long"] = close[long_entry] + params["tp_atr_mult"] * atr[long_entry]

        # --- ATR-based SL/TP for short entries --------------------------------
        short_entry = signals == -1.0
        if short_entry.any():
            df.loc[short_entry, "bb_stop_short"] = close[short_entry] + params["stop_atr_mult"] * atr[short_entry]
            df.loc[short_entry, "bb_tp_short"] = close[short_entry] - params["tp_atr_mult"] * atr[short_entry]

        return signals