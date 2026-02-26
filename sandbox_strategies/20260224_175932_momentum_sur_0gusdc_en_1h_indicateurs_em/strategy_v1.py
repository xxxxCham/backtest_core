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
                max_val=8.0,
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Compute EMA values
        ema_fast_vals = ema_fast
        ema_slow_vals = ema_slow

        # Define EMA masks
        ema_fast_long = ema_fast_vals
        ema_slow_long = ema_slow_vals

        # Compute previous EMA values for crossover detection
        prev_ema_fast = np.roll(ema_fast_vals, 1)
        prev_ema_slow = np.roll(ema_slow_vals, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan

        # Long entry conditions
        cross_up_ema = (ema_fast_vals > ema_slow_vals) & (prev_ema_fast <= prev_ema_slow)
        close_below_bb = close < indicators['bollinger']['middle']
        volume_positive = volume_osc > 0

        long_mask = cross_up_ema & close_below_bb & volume_positive

        # Short entry conditions
        cross_down_ema = (ema_fast_vals < ema_slow_vals) & (prev_ema_fast >= prev_ema_slow)
        close_above_bb = close > indicators['bollinger']['middle']
        volume_negative = volume_osc < 0

        short_mask = cross_down_ema & close_above_bb & volume_negative

        # Exit conditions
        exit_long = cross_down_ema | (close < indicators['bollinger']['lower']) | (volume_osc < 0)
        exit_short = cross_up_ema | (close > indicators['bollinger']['upper']) | (volume_osc > 0)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exit signals
        signals[exit_long & (signals == 1.0)] = 0.0
        signals[exit_short & (signals == -1.0)] = 0.0

        # Set SL/TP levels for long entries
        entry_long_mask = (signals == 1.0)
        if entry_long_mask.any():
            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # Set SL/TP levels for short entries
        entry_short_mask = (signals == -1.0)
        if entry_short_mask.any():
            df.loc[:, "bb_stop_short"] = np.nan
            df.loc[:, "bb_tp_short"] = np.nan
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals