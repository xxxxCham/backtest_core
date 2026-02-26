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
        return ['ema', 'bollinger', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vix_threshold': 70,
         'volume_oscillator_period': 10,
         'volume_threshold': 1.5,
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
            'volume_threshold': ParameterSpec(
                name='volume_threshold',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # compute EMA crossover signals
        fast = ema_fast
        slow = ema_slow
        prev_fast = np.roll(fast, 1)
        prev_slow = np.roll(slow, 1)
        prev_fast[0] = np.nan
        prev_slow[0] = np.nan
        cross_up = (fast > slow) & (prev_fast <= prev_slow)
        cross_down = (fast < slow) & (prev_fast >= prev_slow)
        # compute bollinger band contraction
        bb_width = indicators['bollinger']['upper'] - indicators['bollinger']['lower']
        prev_bb_width = np.roll(bb_width, 1)
        prev_bb_width[0] = np.nan
        bb_contracting = bb_width < prev_bb_width
        # compute volume threshold
        vol_avg = np.nanmean(volume_osc)
        vol_threshold = params.get("volume_threshold", 1.5) * vol_avg
        vol_spike = volume_osc > vol_threshold
        # entry conditions
        long_entry = cross_up & bb_contracting & vol_spike
        short_entry = cross_down & bb_contracting & vol_spike
        long_mask[long_entry] = True
        short_mask[short_entry] = True
        # exit conditions
        momentum = np.diff(fast)
        momentum = np.insert(momentum, 0, 0.0)
        momentum = np.nan_to_num(momentum)
        momentum_neg = momentum < 0
        momentum_neg_3 = momentum_neg & np.roll(momentum_neg, 1) & np.roll(momentum_neg, 2)
        # exit on momentum loss or bollinger middle band cross
        exit_long = momentum_neg_3 | (df["close"] < indicators['bollinger']['middle'])
        exit_short = momentum_neg_3 | (df["close"] > indicators['bollinger']['middle'])
        # apply exits
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0
        # apply entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = df.loc[entry_long, "close"] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = df.loc[entry_long, "close"] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = df.loc[entry_short, "close"] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = df.loc[entry_short, "close"] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
