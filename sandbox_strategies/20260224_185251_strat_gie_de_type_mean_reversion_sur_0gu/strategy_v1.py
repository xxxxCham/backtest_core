from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_inverse_mode')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'supertrend', 'ema', 'atr', 'pivot_points']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 2.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=2.0,
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
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vt = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vt["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vt["vi_minus"])
        vi_signal = np.nan_to_num(vt["signal"])
        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_upper_bb[0] = np.nan
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_lower_bb[0] = np.nan
        cross_up_bb = (close > upper_bb) & (prev_close <= prev_upper_bb)
        cross_down_bb = (close < lower_bb) & (prev_close >= prev_lower_bb)
        vortex_cross_up = (indicators['vortex']['vi_plus'] < vi_signal) & (np.roll(indicators['vortex']['vi_plus'], 1) >= np.roll(vi_signal, 1))
        vortex_cross_down = (indicators['vortex']['vi_minus'] > vi_signal) & (np.roll(indicators['vortex']['vi_minus'], 1) <= np.roll(vi_signal, 1))
        long_condition = cross_up_bb & vortex_cross_up & (direction == 1)
        short_condition = cross_down_bb & vortex_cross_down & (direction == -1)
        long_mask = long_condition
        short_mask = short_condition
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        entry_long_mask = long_mask
        entry_short_mask = short_mask
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        exit_long_mask = (close < ema) | (close > r1)
        exit_short_mask = (close > ema) | (close < s1)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals