from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_0gusdc_1h')

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
         'tp_atr_mult': 2.0,
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

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])  # This is a single array
        ema_slow = np.nan_to_num(indicators['ema'])  # This is a single array
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])

        # EMA crossover logic
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        ema_cross_up = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)

        # Bollinger expansion/contraction
        bb_width = indicators['bollinger']['upper'] - indicators['bollinger']['lower']
        prev_bb_width = np.roll(bb_width, 1)
        prev_bb_width[0] = np.nan
        bb_expansion = bb_width > prev_bb_width
        bb_contraction = bb_width < prev_bb_width

        # Volume oscillator condition
        vol_positive = volume_osc > 0
        vol_negative = volume_osc < 0

        # Session filter (1h-1h UTC)
        # Assume df has a datetime index with UTC timezone
        hour = df.index.hour
        session_filter = (hour >= 1) & (hour <= 1)

        # Entry conditions
        long_entry = ema_cross_up & bb_expansion & vol_positive & session_filter
        short_entry = ema_cross_down & bb_contraction & vol_negative & session_filter

        # Confirm long entry with return to upper band
        bb_upper_violation = df["close"] > indicators['bollinger']['upper']
        prev_bb_upper_violation = np.roll(bb_upper_violation, 1)
        prev_bb_upper_violation[0] = False
        long_confirm = (bb_upper_violation & ~prev_bb_upper_violation) & (df["close"] < indicators['bollinger']['upper'])

        # Confirm short entry with return to lower band
        bb_lower_violation = df["close"] < indicators['bollinger']['lower']
        prev_bb_lower_violation = np.roll(bb_lower_violation, 1)
        prev_bb_lower_violation[0] = False
        short_confirm = (bb_lower_violation & ~prev_bb_lower_violation) & (df["close"] > indicators['bollinger']['lower'])

        # Apply confirmations
        long_mask = long_entry & long_confirm
        short_mask = short_entry & short_confirm

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management
        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_mask_long = signals == 1.0
        entry_mask_short = signals == -1.0

        close = df["close"].values
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        df.loc[entry_mask_long, "bb_stop_long"] = close[entry_mask_long] - stop_atr_mult * atr[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = close[entry_mask_long] + tp_atr_mult * atr[entry_mask_long]

        df.loc[entry_mask_short, "bb_stop_short"] = close[entry_mask_short] + stop_atr_mult * atr[entry_mask_short]
        df.loc[entry_mask_short, "bb_tp_short"] = close[entry_mask_short] - tp_atr_mult * atr[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals
