from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_keltner_adx_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 1.5,
         'tp_atr_mult_trend': 3.0,
         'warmup': 50}

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
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # unwrap indicators
        close = df["close"].values
        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        lower = np.nan_to_num(kelt["lower"])
        middle = np.nan_to_num(kelt["middle"])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # helper cross functions
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

        # Entry conditions
        long_mask = cross_up & (adx_arr > 25.0)
        short_mask = cross_down & (adx_arr > 25.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        cross_middle_up = (close > middle) & (prev_close <= prev_middle)
        cross_middle_down = (close < middle) & (prev_close >= prev_middle)
        cross_middle = cross_middle_up | cross_middle_down
        exit_mask = cross_middle | (adx_arr < 20.0)
        signals[exit_mask] = 0.0

        # warmup
        signals.iloc[:warmup] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = (
                close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            )
            df.loc[long_mask, "bb_tp_long"] = (
                close[long_mask] + params["tp_atr_mult_trend"] * atr[long_mask]
            )

        # Short entry SL/TP
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = (
                close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            )
            df.loc[short_mask, "bb_tp_short"] = (
                close[short_mask] - params["tp_atr_mult_trend"] * atr[short_mask]
            )
        signals.iloc[:warmup] = 0.0
        return signals
