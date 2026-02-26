from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_0gusdc_1h_ema_bollinger_volume_inverse')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
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
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=40,
                default=20,
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
            'volume_oscillator_slow': ParameterSpec(
                name='volume_oscillator_slow',
                min_val=20,
                max_val=50,
                default=26,
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
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bollinger = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(indicators['bollinger']["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(indicators['bollinger']["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(indicators['bollinger']["lower"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        adx = indicators['adx']
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        # compute ema values
        ema_fast_val = ema_fast
        ema_slow_val = ema_slow
        # compute previous values for crossovers
        prev_ema_fast = np.roll(ema_fast_val, 1)
        prev_ema_slow = np.roll(ema_slow_val, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        # long entry conditions
        ema_cross_up = (ema_fast_val > ema_slow_val) & (prev_ema_fast <= prev_ema_slow)
        close_inside_bb = (df["close"].values > indicators['bollinger']['lower']) & (df["close"].values < indicators['bollinger']['upper'])
        vol_positive = volume_oscillator > 0.5
        # short entry conditions
        ema_cross_down = (ema_fast_val < ema_slow_val) & (prev_ema_fast >= prev_ema_slow)
        close_inside_bb_short = (df["close"].values > indicators['bollinger']['lower']) & (df["close"].values < indicators['bollinger']['upper'])
        vol_negative = volume_oscillator < -0.5
        # adx filter for trend
        adx_threshold = 20.0
        adx_trend = adx_val > adx_threshold
        # long signals
        long_condition = ema_cross_up & close_inside_bb & vol_positive & adx_trend
        long_mask = long_condition
        # short signals
        short_condition = ema_cross_down & close_inside_bb_short & vol_negative & adx_trend
        short_mask = short_condition
        # exit conditions
        exit_long = (ema_fast_val < ema_slow_val) | (df["close"].values > indicators['bollinger']['upper']) | (df["close"].values < indicators['bollinger']['lower'])
        exit_short = (ema_fast_val > ema_slow_val) | (df["close"].values > indicators['bollinger']['upper']) | (df["close"].values < indicators['bollinger']['lower'])
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP levels for long positions
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan
            close = df["close"].values
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # set SL/TP levels for short positions
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[:, "bb_stop_short"] = np.nan
            df.loc[:, "bb_tp_short"] = np.nan
            close = df["close"].values
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
