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
         'bollinger_std_dev': 2,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        # extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        atr = np.nan_to_num(indicators['atr'])
        # entry conditions
        # long entry: close crosses above bb.upper, ema > bb.middle, vol_osc > 0, supertrend direction < 1
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        cross_above_bb = (close > indicators['bollinger']['upper']) & (prev_close <= prev_bb_upper)
        long_condition = (ema > indicators['bollinger']['middle']) & (vol_osc > 0) & (st_direction < 1)
        long_mask = cross_above_bb & long_condition
        # short entry: close crosses below bb.lower, ema < bb.middle, vol_osc < 0, supertrend direction > 1
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_bb_lower[0] = np.nan
        cross_below_bb = (close < indicators['bollinger']['lower']) & (prev_close >= prev_bb_lower)
        short_condition = (ema < indicators['bollinger']['middle']) & (vol_osc < 0) & (st_direction > 1)
        short_mask = cross_below_bb & short_condition
        # exit conditions
        # exit long: close crosses below bb.middle
        prev_bb_middle = np.roll(indicators['bollinger']['middle'], 1)
        prev_bb_middle[0] = np.nan
        exit_long = (close < indicators['bollinger']['middle']) & (prev_close >= prev_bb_middle)
        # exit short: close crosses above bb.middle
        exit_short = (close > indicators['bollinger']['middle']) & (prev_close <= prev_bb_middle)
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # apply exits
        exit_long_mask = signals == 1.0
        exit_long_mask = exit_long_mask & exit_long
        signals[exit_long_mask] = 0.0
        exit_short_mask = signals == -1.0
        exit_short_mask = exit_short_mask & exit_short
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
