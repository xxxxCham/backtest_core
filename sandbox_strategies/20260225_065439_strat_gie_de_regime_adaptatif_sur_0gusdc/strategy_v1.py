from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_keltner_vwap_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 3.2,
         'vol_factor': 1.0,
         'vwap_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_mult': ParameterSpec(
                name='keltner_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'vwap_period': ParameterSpec(
                name='vwap_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'vol_factor': ParameterSpec(
                name='vol_factor',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=5.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=3.2,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=10,
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
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        close = df["close"].values

        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])

        vwap = np.nan_to_num(indicators['vwap'])
        atr = np.nan_to_num(indicators['atr'])

        vol_factor = float(params.get("vol_factor", 1.0))

        # Regime determination
        width = indicators['keltner']['upper'] - indicators['keltner']['lower']
        regime_high = width > atr * vol_factor
        regime_low = ~regime_high

        # Entry conditions
        long_breakout = regime_high & (close > indicators['keltner']['upper'])
        long_meanrev = regime_low & (close > vwap)
        long_mask = long_breakout | long_meanrev

        short_breakout = regime_high & (close < indicators['keltner']['lower'])
        short_meanrev = regime_low & (close < vwap)
        short_mask = short_breakout | short_meanrev

        # Exit condition: regime change (cross of width vs atr*vol_factor)
        threshold = atr * vol_factor
        prev_width = np.roll(width, 1)
        prev_thr = np.roll(threshold, 1)
        prev_width[0] = np.nan
        prev_thr[0] = np.nan
        cross_up = (width > threshold) & (prev_width <= prev_thr)
        cross_down = (width < threshold) & (prev_width >= prev_thr)
        exit_mask = cross_up | cross_down

        # Apply exit mask (force flat)
        signals[exit_mask] = 0.0

        # Apply entry masks, ensuring exit overrides entry
        long_entry = long_mask & ~exit_mask
        short_entry = short_mask & ~exit_mask

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # ATR‑based SL/TP levels
        stop_mult = float(params.get("stop_atr_mult", 1.1))
        tp_mult = float(params.get("tp_atr_mult", 3.2))

        # Long SL/TP
        if long_entry.any():
            df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
            df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]

        # Short SL/TP
        if short_entry.any():
            df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
            df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
