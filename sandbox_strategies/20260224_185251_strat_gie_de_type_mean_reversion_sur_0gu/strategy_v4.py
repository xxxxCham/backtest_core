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
        return ['bollinger', 'vortex', 'supertrend', 'atr', 'ema', 'pivot_points']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.0,
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
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.0,
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
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vx = indicators['vortex']
        vi_plus = np.nan_to_num(vx["vi_plus"])
        vi_minus = np.nan_to_num(vx["vi_minus"])
        vi_signal = np.nan_to_num(vx["signal"])
        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])
        close = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators['atr'])
        ema = np.nan_to_num(indicators['ema'])
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        # Define range regime: Supertrend direction = 1 (trend up) or -1 (trend down)
        # For range, we want direction to be 0 (flat), but since Supertrend is binary,
        # we consider regime flat when direction is 0 (i.e., not in trend)
        is_range = (direction == 0)
        # Entry conditions
        # Long entry: price touches upper BB AND vortex crosses below signal AND is in range
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_upper_bb[0] = np.nan
        touch_upper = (close >= upper_bb) & (prev_upper_bb < upper_bb)
        vortex_cross_down = (vi_plus < vi_signal) & (np.roll(vi_plus, 1) >= vi_signal)
        long_condition = touch_upper & vortex_cross_down & is_range
        long_mask = long_condition
        # Short entry: price touches lower BB AND vortex crosses above signal AND is in range
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_lower_bb[0] = np.nan
        touch_lower = (close <= lower_bb) & (prev_lower_bb > lower_bb)
        vortex_cross_up = (vi_plus > vi_signal) & (np.roll(vi_plus, 1) <= vi_signal)
        short_condition = touch_lower & vortex_cross_up & is_range
        short_mask = short_condition
        # Exit conditions
        # Exit long on EMA cross
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        exit_long = (close < ema) & (prev_ema >= ema)
        # Exit short on EMA cross
        exit_short = (close > ema) & (prev_ema <= ema)
        # Combine exit conditions
        exit_long_mask = exit_long
        exit_short_mask = exit_short
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Long entry SL/TP
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = r1[entry_long_mask]
        # Short entry SL/TP
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = r1[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals