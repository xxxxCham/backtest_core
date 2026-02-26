from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_zigzag_volume_contrarian')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'pivot_points', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'pivot_period': 20,
         'stop_atr_mult': 2.0,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=1,
                max_val=5,
                default=3,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
                default=14,
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

        # Get indicator data
        st = indicators['supertrend']
        st_direction = st["direction"].astype(np.float64)
        pp = indicators['pivot_points']
        r1 = pp["r1"].astype(np.float64)
        s1 = pp["s1"].astype(np.float64)
        atr = indicators['atr'].astype(np.float64)
        close = df["close"].values

        # Calculate SMA of ATR for volatility filter
        atr_sma_period = 20
        atr_sma = np.convolve(atr, np.ones(atr_sma_period)/atr_sma_period, mode='same')
        atr_sma = np.nan_to_num(atr_sma)

        # Create cross helper functions
        def cross_up(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = x[0]
            prev_y[0] = y[0]
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = x[0]
            prev_y[0] = y[0]
            return (x < y) & (prev_x >= prev_y)

        # Detect SuperTrend direction changes
        prev_st_direction = np.roll(st_direction, 1)
        prev_st_direction[0] = st_direction[0]

        st_trend_up = st_direction > 0
        st_trend_down = st_direction <= 0
        prev_st_trend_up = prev_st_direction > 0
        prev_st_trend_down = prev_st_direction <= 0

        # Long entry: SuperTrend changes from downtrend to uptrend AND close crosses above R1 AND ATR > SMA(ATR, 20)
        st_bullish_change = st_trend_up & prev_st_trend_down
        close_above_r1 = cross_up(close, r1)
        volatility_filter = atr > atr_sma
        long_entry = st_bullish_change & close_above_r1 & volatility_filter

        # Short entry: SuperTrend changes from uptrend to downtrend AND close crosses below S1 AND ATR > SMA(ATR, 20)
        st_bearish_change = st_trend_down & prev_st_trend_up
        close_below_s1 = cross_down(close, s1)
        short_entry = st_bearish_change & close_below_s1 & volatility_filter

        # Exit conditions
        # Long exit: SuperTrend direction reverses OR close crosses below S1
        long_exit = (st_trend_down & prev_st_trend_up) | cross_down(close, s1)

        # Short exit: SuperTrend direction reverses OR close crosses above R1
        short_exit = (st_trend_up & prev_st_trend_down) | cross_up(close, r1)

        # Apply signals with warmup protection
        valid_mask = np.arange(n) >= warmup
        long_mask = long_entry & valid_mask
        short_mask = short_entry & valid_mask

        # Apply exit conditions
        long_exit_mask = long_exit & valid_mask
        short_exit_mask = short_exit & valid_mask

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR-based stop loss and take profit
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        # Long positions
        long_entries = long_mask & ~long_exit_mask
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_atr_mult * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_atr_mult * atr[long_entries]

        # Short positions
        short_entries = short_mask & ~short_exit_mask
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_atr_mult * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_atr_mult * atr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals