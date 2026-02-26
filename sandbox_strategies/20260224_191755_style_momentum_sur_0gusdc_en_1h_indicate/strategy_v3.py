from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='volume_liquidity_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'obv_period': 20,
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

        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])

        # Compute EMA arrays
        ema_50 = ema_fast
        ema_200 = ema_slow

        # Volume oscillators for long and short
        vol_long = volume_oscillator
        vol_short = volume_oscillator

        # Compute previous values for OBV and Volume Oscillator
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan
        vol_long_prev = np.roll(vol_long, 1)
        vol_long_prev[0] = np.nan
        vol_short_prev = np.roll(vol_short, 1)
        vol_short_prev[0] = np.nan

        # Volume oscillators for 2 and 3 periods ago
        vol_long_prev2 = np.roll(vol_long, 2)
        vol_long_prev2[0] = np.nan
        vol_long_prev2[1] = np.nan
        vol_long_prev3 = np.roll(vol_long, 3)
        vol_long_prev3[0] = np.nan
        vol_long_prev3[1] = np.nan
        vol_long_prev3[2] = np.nan

        vol_short_prev2 = np.roll(vol_short, 2)
        vol_short_prev2[0] = np.nan
        vol_short_prev2[1] = np.nan
        vol_short_prev3 = np.roll(vol_short, 3)
        vol_short_prev3[0] = np.nan
        vol_short_prev3[1] = np.nan
        vol_short_prev3[2] = np.nan

        # Entry conditions
        long_condition = (ema_50 > ema_200) & (obv > obv_prev) & (vol_long > vol_long_prev) & (vol_long > vol_long_prev2) & (vol_long > vol_long_prev3)
        short_condition = (ema_50 < ema_200) & (obv < obv_prev) & (vol_short < vol_short_prev) & (vol_short < vol_short_prev2) & (vol_short < vol_short_prev3)

        long_mask[long_condition] = True
        short_mask[short_condition] = True

        # Exit conditions
        exit_long = (ema_50 < ema_200) | (vol_long < vol_long_prev)
        exit_short = (ema_50 > ema_200) | (vol_short < vol_short_prev)

        # Apply exits
        exit_long_mask = np.zeros(n, dtype=bool)
        exit_short_mask = np.zeros(n, dtype=bool)
        exit_long_mask[exit_long] = True
        exit_short_mask[exit_short] = True

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Handle exits
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)

        close = df["close"].values
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
