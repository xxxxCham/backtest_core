from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='atr_adx_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'donchian']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'donchian_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 3.0,
         'tp_range_atr_mult': 1.6,
         'tp_trend_atr_mult': 4.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_trend_atr_mult': ParameterSpec(
                name='tp_trend_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'tp_range_atr_mult': ParameterSpec(
                name='tp_range_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.6,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        # extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx = np.nan_to_num(indicators['adx']["adx"])

        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        middle = np.nan_to_num(dc["middle"])
        lower = np.nan_to_num(dc["lower"])

        # cross helpers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan

        cross_up = (close > upper) & (prev_close <= prev_upper)
        cross_down = (close < lower) & (prev_close >= prev_lower)

        long_mask = cross_up & (adx > 25)
        short_mask = cross_down & (adx > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions (not directly used in signals but kept for reference)
        exit_cross = ((close < middle) & (prev_close >= prev_middle)) | (adx < 20)
        signals[exit_cross] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute ATR-based levels on entry bars
        entry_long = long_mask
        entry_short = short_mask

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_trend_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_trend_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
