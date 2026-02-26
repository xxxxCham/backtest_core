from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_keltner_adx_30m_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'band_width_atr_mult': 1.5,
         'leverage': 1,
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
                min_val=2.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'band_width_atr_mult': ParameterSpec(
                name='band_width_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=5,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        middle = np.nan_to_num(kelt["middle"])
        lower = np.nan_to_num(kelt["lower"])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])

        band_width = upper - lower
        band_cond = band_width > params["band_width_atr_mult"] * atr
        adx_cond = adx_arr > 25

        # cross helper
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

        long_mask = cross_up & adx_cond & band_cond
        short_mask = cross_down & adx_cond & band_cond

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic: cross middle or adx < 20
        exit_cross_mid_long = (close <= middle) & (prev_close > prev_middle)
        exit_cross_mid_short = (close >= middle) & (prev_close < prev_middle)
        exit_adx = adx_arr < 20

        exit_mask = (signals == 1.0) & (exit_cross_mid_long | exit_adx)
        signals[exit_mask] = 0.0
        exit_mask_short = (signals == -1.0) & (exit_cross_mid_short | exit_adx)
        signals[exit_mask_short] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = long_mask
        entry_short = short_mask

        stop_long = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        tp_long_trend = close[entry_long] + params["tp_atr_mult_trend"] * atr[entry_long]
        tp_long_range = close[entry_long] + params["tp_atr_mult_range"] * atr[entry_long]
        tp_long = np.where(adx_arr[entry_long] > 25, tp_long_trend, tp_long_range)

        df.loc[entry_long, "bb_stop_long"] = stop_long
        df.loc[entry_long, "bb_tp_long"] = tp_long

        stop_short = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        tp_short_trend = close[entry_short] - params["tp_atr_mult_trend"] * atr[entry_short]
        tp_short_range = close[entry_short] - params["tp_atr_mult_range"] * atr[entry_short]
        tp_short = np.where(adx_arr[entry_short] > 25, tp_short_trend, tp_short_range)

        df.loc[entry_short, "bb_stop_short"] = stop_short
        df.loc[entry_short, "bb_tp_short"] = tp_short
        signals.iloc[:warmup] = 0.0
        return signals
