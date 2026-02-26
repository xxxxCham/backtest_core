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
        return ['ema', 'bollinger', 'volume_oscillator', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
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
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])
        # compute EMA crossover signals
        ema_fast_vals = ema_fast
        ema_slow_vals = ema_slow
        prev_ema_fast = np.roll(ema_fast_vals, 1)
        prev_ema_slow = np.roll(ema_slow_vals, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        ema_cross_up = (ema_fast_vals > ema_slow_vals) & (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast_vals < ema_slow_vals) & (prev_ema_fast >= prev_ema_slow)
        # compute Bollinger crossover signals
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        bb_lower_cross_up = (close > indicators['bollinger']['lower']) & (prev_close <= indicators['bollinger']['lower'])
        bb_upper_cross_down = (close < indicators['bollinger']['upper']) & (prev_close >= indicators['bollinger']['upper'])
        # define long and short entry conditions
        long_entry = (
            ema_cross_up &
            bb_lower_cross_up &
            (volume_osc > 0) &
            (rsi > params["rsi_oversold"])
        )
        short_entry = (
            ema_cross_down &
            bb_upper_cross_down &
            (volume_osc < 0) &
            (rsi < params["rsi_overbought"])
        )
        # define exit conditions
        long_exit = ema_cross_down | (close > indicators['bollinger']['upper']) | (rsi > params["rsi_overbought"])
        short_exit = ema_cross_up | (close < indicators['bollinger']['lower']) | (rsi < params["rsi_oversold"])
        # initialize masks
        long_mask = long_entry
        short_mask = short_entry
        # apply exit conditions
        # for longs, we need to identify when to exit (cross down or hit upper band)
        long_exit_mask = long_exit
        # for shorts, we need to identify when to exit (cross up or hit lower band)
        short_exit_mask = short_exit
        # mask longs to avoid double entry
        long_mask = long_mask & ~long_exit_mask
        short_mask = short_mask & ~short_exit_mask
        # apply long and short signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # apply ATR-based SL and TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
