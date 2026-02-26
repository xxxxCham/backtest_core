from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_ema_bollinger_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
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
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
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
        close = np.nan_to_num(df["close"].values)
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bollinger = indicators['bollinger']
        upper = np.nan_to_num(indicators['bollinger']["upper"])
        middle = np.nan_to_num(indicators['bollinger']["middle"])
        lower = np.nan_to_num(indicators['bollinger']["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # prepare crossover conditions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_slow[0] = np.nan
        cross_above = (close > upper) & (prev_close <= prev_upper)
        cross_below = (close < lower) & (prev_close >= prev_lower)
        # entry conditions
        long_condition = cross_above & (ema_fast > ema_slow) & (volume_osc > 0)
        short_condition = cross_below & (ema_fast < ema_slow) & (volume_osc < 0)
        long_mask = long_condition
        short_mask = short_condition
        # exit conditions
        exit_long = (close < lower) | ((ema_fast < ema_slow) & (volume_osc < 0))
        exit_short = (close > lower) | ((ema_fast > ema_slow) & (volume_osc > 0))
        # handle exits
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
