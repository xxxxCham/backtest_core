from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_regime_keltner_supertrend_bollinger')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'supertrend', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'bollinger_period': 20,
            'bollinger_std_dev': 2.0,
            'keltner_multiplier': 1.5,
            'keltner_period': 20,
            'leverage': 1,
            'stop_atr_mult': 1.6,
            'supertrend_multiplier': 3.0,
            'supertrend_period': 10,
            'tp_atr_mult': 3.0,
            'tp_atr_mult_range': 2.0,
            'tp_atr_mult_trend': 3.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_multiplier': ParameterSpec(
                name='keltner_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=2.0,
                max_val=6.0,
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=3,
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

        # Initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Price series
        close = df["close"].values

        # ---- Indicators (sanitize) ----
        kelt = indicators['keltner']
        # Ensure float dtype for NaN handling
        kelt_upper = np.nan_to_num(kelt["upper"]).astype(float)
        kelt_lower = np.nan_to_num(kelt["lower"]).astype(float)

        st = indicators['supertrend']
        # Convert direction to float to allow NaN later
        st_dir = st["direction"].astype(float)

        bb = indicators['bollinger']
        bb_upper = np.nan_to_num(bb["upper"]).astype(float)
        bb_lower = np.nan_to_num(bb["lower"]).astype(float)

        atr = np.nan_to_num(indicators['atr']).astype(float)

        # ---- Entry conditions ----
        long_entry = (close > kelt_upper) & (st_dir == 1) & (close > bb_upper)
        short_entry = (close < kelt_lower) & (st_dir == -1) & (close < bb_lower)

        long_mask[long_entry] = True
        short_mask[short_entry] = True

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ---- Risk management ----
        stop_atr_mult = float(params.get("stop_atr_mult", 1.6))
        tp_atr_mult_trend = float(params.get("tp_atr_mult_trend", 3.5))
        tp_atr_mult_range = float(params.get("tp_atr_mult_range", 2.0))

        tp_mult = tp_atr_mult_trend  # trend confirmed by entry logic

        # Initialise SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        # ---- Exit conditions ----
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        prev_bb_upper = np.roll(bb_upper, 1)
        prev_bb_upper[0] = np.nan

        prev_bb_lower = np.roll(bb_lower, 1)
        prev_bb_lower[0] = np.nan

        cross_down_lower = (close < bb_lower) & (prev_close >= prev_bb_lower)
        cross_up_upper = (close > bb_upper) & (prev_close <= prev_bb_upper)

        # Supertrend direction change (use float array to allow NaN)
        prev_st_dir = np.roll(st_dir, 1).astype(float)
        prev_st_dir[0] = np.nan
        direction_change = st_dir != prev_st_dir

        # Combine exit masks (engine will handle flatting)
        _exit_long = cross_down_lower | direction_change
        _exit_short = cross_up_upper | direction_change

        # No modification of signals needed; they remain 0 on exit bars
        signals.iloc[:warmup] = 0.0
        return signals