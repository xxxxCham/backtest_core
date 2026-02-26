from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_macd_adx_atr_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.2, 'tp_atr_mult': 5.9, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.9,
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

        # unwrap indicators
        st_line = np.nan_to_num(indicators['supertrend']["supertrend"])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # entry conditions
        long_mask = (close > st_line) & (macd_hist > 0) & (adx_val > 25)
        short_mask = (close < st_line) & (macd_hist < 0) & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        cross_down_hist = (macd_hist < 0) & (prev_hist >= 0)
        cross_up_hist = (macd_hist > 0) & (prev_hist <= 0)

        exit_long_mask = cross_down_hist | (close < st_line) | (adx_val < 20)
        exit_short_mask = cross_up_hist | (close > st_line) | (adx_val < 20)

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # avoid consecutive identical signals
        prev_signals = np.roll(signals.values, 1)
        prev_signals[0] = 0.0
        duplicate_mask = signals.values == prev_signals
        signals[duplicate_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute SL/TP on entry bars
        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]

        signals.iloc[:warmup] = 0.0
        return signals