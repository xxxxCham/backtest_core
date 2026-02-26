from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_sur_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Define cross helpers
        prev_ema = np.roll(ema, 1)
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_vol_osc = np.roll(vol_osc, 1)
        prev_st_direction = np.roll(st_direction, 1)
        prev_ema[0] = 0.0
        prev_bb_upper[0] = 0.0
        prev_bb_lower[0] = 0.0
        prev_vol_osc[0] = 0.0
        prev_st_direction[0] = 0.0

        # Entry conditions
        # Long entry: close crosses above ema, ema crosses above bb.upper, volume positive, supertrend up
        ema_cross_up_bb = (ema > indicators['bollinger']['upper']) & (prev_ema <= prev_bb_upper)
        vol_positive = vol_osc > 0
        trend_up = st_direction > 0
        close_cross_above_ema = (close > ema) & (np.roll(close, 1) <= prev_ema)
        long_entry = close_cross_above_ema & ema_cross_up_bb & vol_positive & trend_up

        # Short entry: close crosses below ema, ema crosses below bb.lower, volume negative, supertrend down
        ema_cross_down_bb = (ema < indicators['bollinger']['lower']) & (prev_ema >= prev_bb_lower)
        vol_negative = vol_osc < 0
        trend_down = st_direction < 0
        close_cross_below_ema = (close < ema) & (np.roll(close, 1) >= prev_ema)
        short_entry = close_cross_below_ema & ema_cross_down_bb & vol_negative & trend_down

        # Exit conditions
        # Exit long if close crosses below ema
        exit_long = (close < ema) & (np.roll(close, 1) >= prev_ema)
        # Exit short if close crosses above ema
        exit_short = (close > ema) & (np.roll(close, 1) <= prev_ema)

        # Set masks
        long_mask = long_entry
        short_mask = short_entry

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exit signals
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Write SL/TP columns for ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long SL/TP
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        # Short SL/TP
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals