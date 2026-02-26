from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_bollinger_volume_momentum')

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
         'tp_atr_mult': 2.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
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
            'volume_oscillator_short': ParameterSpec(
                name='volume_oscillator_short',
                min_val=5,
                max_val=30,
                default=12,
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
        signals.iloc[:warmup] = 0.0
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        close = df["close"].values
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        bollinger = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(indicators['bollinger']["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(indicators['bollinger']["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(indicators['bollinger']["lower"])
        ema_fast = ema_fast.reshape(-1)
        ema_slow = ema_slow.reshape(-1)
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        cross_up_ema = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        cross_down_ema = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)
        inside_bb = (close > indicators['bollinger']['lower']) & (close < indicators['bollinger']['upper'])
        volume_filter = volume_oscillator > 0.5
        long_entry = cross_up_ema & inside_bb & volume_filter
        short_entry = cross_down_ema & inside_bb & volume_filter
        long_mask = long_entry
        short_mask = short_entry
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        exit_long = cross_down_ema | (close >= indicators['bollinger']['upper'])
        exit_short = cross_up_ema | (close <= indicators['bollinger']['lower'])
        exit_mask_long = exit_long & (np.roll(signals, 1) == 1.0)
        exit_mask_short = exit_short & (np.roll(signals, 1) == -1.0)
        signals[exit_mask_long] = 0.0
        signals[exit_mask_short] = 0.0
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_mask_long = (signals == 1.0)
        entry_mask_short = (signals == -1.0)
        df.loc[entry_mask_long, "bb_stop_long"] = close[entry_mask_long] - params["stop_atr_mult"] * atr[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = close[entry_mask_long] + params["tp_atr_mult"] * atr[entry_mask_long]
        df.loc[entry_mask_short, "bb_stop_short"] = close[entry_mask_short] + params["stop_atr_mult"] * atr[entry_mask_short]
        df.loc[entry_mask_short, "bb_tp_short"] = close[entry_mask_short] - params["tp_atr_mult"] * atr[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals
