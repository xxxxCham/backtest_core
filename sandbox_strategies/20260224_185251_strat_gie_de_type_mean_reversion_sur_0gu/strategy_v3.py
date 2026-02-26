from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_inverse_mode_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'supertrend', 'ema', 'atr', 'pivot_points']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
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
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vt = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vt["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vt["vi_minus"])
        signal_vt = np.nan_to_num(vt["signal"])
        st = indicators['supertrend']
        direction_st = np.nan_to_num(st["direction"])
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        r2 = np.nan_to_num(pp["r2"])
        close = df["close"].values

        # Entry conditions
        # Long entry: price touches upper band AND vortex indicators['vortex']['vi_plus'] crosses above signal AND supertrend direction is 1
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_upper_bb[0] = np.nan
        price_touch_upper = (close == upper_bb)
        cross_up_vortex = (indicators['vortex']['vi_plus'] > signal_vt) & (np.roll(indicators['vortex']['vi_plus'], 1) <= np.roll(signal_vt, 1))
        regime_filter_long = (direction_st == 1)
        long_entry = price_touch_upper & cross_up_vortex & regime_filter_long

        # Short entry: price touches lower band AND vortex indicators['vortex']['vi_minus'] crosses below signal AND supertrend direction is -1
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_lower_bb[0] = np.nan
        price_touch_lower = (close == lower_bb)
        cross_down_vortex = (indicators['vortex']['vi_minus'] < signal_vt) & (np.roll(indicators['vortex']['vi_minus'], 1) >= np.roll(signal_vt, 1))
        regime_filter_short = (direction_st == -1)
        short_entry = price_touch_lower & cross_down_vortex & regime_filter_short

        # Set masks
        long_mask = long_entry
        short_mask = short_entry

        # Exit conditions
        # Exit long: price crosses below EMA or price crosses above R2
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        exit_long_ema = (close < ema) & (prev_ema >= ema)
        exit_long_r2 = (close > r2) & (np.roll(close, 1) <= r2)
        exit_long = exit_long_ema | exit_long_r2

        # Exit short: price crosses above EMA or price crosses below R2
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        exit_short_ema = (close > ema) & (prev_ema <= ema)
        exit_short_r2 = (close < r2) & (np.roll(close, 1) >= r2)
        exit_short = exit_short_ema | exit_short_r2

        # Apply exits to existing positions
        # Long exits
        long_exit_mask = exit_long
        short_exit_mask = exit_short

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # Risk management: write SL/TP levels into df
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
