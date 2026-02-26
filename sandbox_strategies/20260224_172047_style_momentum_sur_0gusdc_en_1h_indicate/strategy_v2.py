from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_volume_williams_momentum_revised')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_period': 20,
         'warmup': 50,
         'williams_r_overbought': -20,
         'williams_r_oversold': -80,
         'williams_r_period': 14}

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
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=30,
                default=14,
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

        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])

        fast = ema_fast
        slow = ema_slow

        # EMA crossover signals
        prev_fast = np.roll(fast, 1)
        prev_slow = np.roll(slow, 1)
        prev_fast[0] = np.nan
        prev_slow[0] = np.nan

        cross_up = (fast > slow) & (prev_fast <= prev_slow)
        cross_down = (fast < slow) & (prev_fast >= prev_slow)

        # Volume condition
        volume_sma = np.nanmean(volume_oscillator)
        volume_condition = volume_oscillator > volume_sma

        # Entry conditions
        long_entry = cross_up & volume_condition & (williams_r < params["williams_r_oversold"])
        short_entry = cross_down & volume_condition & (williams_r > params["williams_r_overbought"])

        # Exit conditions
        long_exit = cross_down | (williams_r > params["williams_r_overbought"])
        short_exit = cross_up | (williams_r < params["williams_r_oversold"])

        # Volatility filter
        volatility_filter = atr > 0.001

        # Apply signals
        long_mask = long_entry & volatility_filter
        short_mask = short_entry & volatility_filter

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit signals
        exit_long = long_exit & (signals == 1.0)
        exit_short = short_exit & (signals == -1.0)

        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Write SL/TP columns if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        close = df["close"].values
        atr_mult = np.nan_to_num(atr)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr_mult[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr_mult[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr_mult[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr_mult[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
