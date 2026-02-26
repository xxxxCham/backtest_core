from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_style_0gusdc_arbusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=100,
                max_val=300,
                default=200,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_fast': ParameterSpec(
                name='volume_oscillator_fast',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
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
        close = df["close"].values
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])

        # EMA fast and slow values
        ema_fast_val = ema_fast
        ema_slow_val = ema_slow

        # Define EMA conditions
        ema_fast_gt_slow = ema_fast_val > ema_slow_val
        ema_fast_lt_slow = ema_fast_val < ema_slow_val

        # Volume oscillator condition
        vol_positive = volume_osc > 0
        vol_negative = volume_osc < 0

        # Cross detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_upper_bb[0] = np.nan
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_lower_bb[0] = np.nan

        # Long entry: close crosses above upper BB with EMA confirmation and volume > 0
        cross_above_bb = (close > upper_bb) & (prev_close <= prev_upper_bb)
        long_condition = cross_above_bb & ema_fast_gt_slow & vol_positive
        long_mask = long_condition

        # Short entry: close crosses below lower BB with EMA confirmation and volume < 0
        cross_below_bb = (close < lower_bb) & (prev_close >= prev_lower_bb)
        short_condition = cross_below_bb & ema_fast_lt_slow & vol_negative
        short_mask = short_condition

        # Exit conditions
        # For longs: exit when close crosses below lower BB or EMA 200 is declining or volume < 0
        prev_ema_slow = np.roll(ema_slow_val, 1)
        prev_ema_slow[0] = np.nan
        ema_slow_decline = ema_slow_val < prev_ema_slow

        exit_long_condition = (close < lower_bb) | ema_slow_decline | (volume_osc < 0)
        exit_long_mask = exit_long_condition

        # For shorts: exit when close crosses above upper BB or EMA 200 is rising or volume > 0
        exit_short_condition = (close > upper_bb) | (~ema_slow_decline) | (volume_osc > 0)
        exit_short_mask = exit_short_condition

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exits
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        entry_long_mask = (signals == 1.0)
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # Short entry SL/TP
        entry_short_mask = (signals == -1.0)
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
