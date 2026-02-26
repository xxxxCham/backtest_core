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
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 3,
         'volume_oscillator_slow': 10,
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
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_fast': ParameterSpec(
                name='volume_oscillator_fast',
                min_val=2,
                max_val=10,
                default=3,
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
        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bollinger = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(indicators['bollinger']["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(indicators['bollinger']["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(indicators['bollinger']["lower"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        # Compute EMA arrays
        ema_fast_vals = ema_fast
        ema_slow_vals = ema_slow
        # Compute volume oscillator
        fast = params["volume_oscillator_fast"]
        slow = params["volume_oscillator_slow"]
        # Volume oscillator comparison
        volume_oscillator_shifted = np.roll(volume_oscillator, fast)
        volume_oscillator_shifted[0:fast] = np.nan
        volume_condition = volume_oscillator > 1.15 * volume_oscillator_shifted
        # EMA crossover
        prev_ema_fast = np.roll(ema_fast_vals, 1)
        prev_ema_slow = np.roll(ema_slow_vals, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        ema_cross_up = (ema_fast_vals > ema_slow_vals) & (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast_vals < ema_slow_vals) & (prev_ema_fast >= prev_ema_slow)
        # Close vs Bollinger bands
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_bb_lower[0] = np.nan
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        close_cross_up_lower = (close > indicators['bollinger']['lower']) & (prev_bb_lower <= indicators['bollinger']['lower'])
        close_cross_down_upper = (close < indicators['bollinger']['upper']) & (prev_bb_upper >= indicators['bollinger']['upper'])
        # Long entry conditions
        long_entry = ema_cross_up & close_cross_up_lower & volume_condition
        long_mask = long_entry
        # Short entry conditions
        short_entry = ema_cross_down & close_cross_down_upper & volume_condition
        short_mask = short_entry
        # Exit conditions
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        exit_long = (close < indicators['bollinger']['upper']) | (ema_slow_vals < close)
        exit_short = (close > indicators['bollinger']['upper']) | (ema_slow_vals > close)
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set SL/TP columns for ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Long SL/TP
        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        # Short SL/TP
        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals