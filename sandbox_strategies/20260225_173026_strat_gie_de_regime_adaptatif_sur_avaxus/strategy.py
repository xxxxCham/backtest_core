from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='avx_30m_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'atr', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'atr_period': 14,
            'keltner_multiplier': 1.5,
            'keltner_period': 20,
            'leverage': 1,
            'stop_atr_mult': 2.0,
            'tp_atr_mult': 5.5,
            'warmup': 50,
            # Thresholds used in the logic
            'adx_trend_threshold': 25,
            'adx_exit_threshold': 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
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
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get('warmup', 50))
        # Boolean masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        middle = np.nan_to_num(kelt["middle"])
        lower = np.nan_to_num(kelt["lower"])

        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])

        # Helper cross functions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan

        cross_up_close_upper = (close > upper) & (prev_close <= prev_upper)
        cross_down_close_lower = (close < lower) & (prev_close >= prev_lower)
        cross_up_close_middle = (close > middle) & (prev_close <= prev_middle)
        cross_down_close_middle = (close < middle) & (prev_close >= prev_middle)

        # Mean‑reversion conditions
        mean_rev_long = (close > middle) & ((close - middle) < atr * 0.5)
        mean_rev_short = (close < middle) & ((middle - close) < atr * 0.5)

        # Thresholds
        trend_threshold = params.get('adx_trend_threshold', 25)
        exit_threshold = params.get('adx_exit_threshold', 20)

        # Entry logic
        long_mask = ((adx_val > trend_threshold) & cross_up_close_upper) | \
                    ((adx_val <= trend_threshold) & mean_rev_long)

        short_mask = ((adx_val > trend_threshold) & cross_down_close_lower) | \
                     ((adx_val <= trend_threshold) & mean_rev_short)

        # Apply entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic
        exit_long_mask = cross_down_close_middle | (adx_val < exit_threshold)
        exit_short_mask = cross_up_close_middle | (adx_val < exit_threshold)

        # Ensure exits do not override entries on the same bar
        exit_long_mask &= ~long_mask
        exit_short_mask &= ~short_mask

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Write SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Long entries SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        # Short entries SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals