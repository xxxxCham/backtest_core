from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_vortex_trend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_avg_period': 20,
         'atr_period': 20,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
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
            'atr_avg_period': ParameterSpec(
                name='atr_avg_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
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

        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        oscillator = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Previous values for crossovers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_upper_bb[0] = np.nan
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_lower_bb[0] = np.nan
        prev_oscillator = np.roll(oscillator, 1)
        prev_oscillator[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above lower bb AND vortex oscillator > 0.5 AND increasing
        long_entry_condition = (
            (close > lower_bb) &
            (prev_close <= prev_lower_bb) &
            (oscillator > 0.5) &
            (oscillator > prev_oscillator)
        )

        # Short entry: close crosses below upper bb AND vortex oscillator > 0.5 AND increasing
        short_entry_condition = (
            (close < upper_bb) &
            (prev_close >= prev_upper_bb) &
            (oscillator > 0.5) &
            (oscillator > prev_oscillator)
        )

        long_mask = long_entry_condition
        short_mask = short_entry_condition

        # Exit conditions
        # Close crosses above upper bb OR close crosses below lower bb OR vortex decreasing
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        exit_long_condition = (
            (close > upper_bb) |
            (prev_close <= prev_upper_bb) |
            (oscillator < prev_oscillator)
        )

        exit_short_condition = (
            (close < lower_bb) |
            (prev_close >= prev_lower_bb) |
            (oscillator < prev_oscillator)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute dynamic SL/TP
        atr_avg = np.nanmean(atr)
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        # Fix: Compare each element of atr with atr_avg using a boolean mask
        atr_condition = atr > atr_avg
        if atr_condition.any():  # Use .any() to avoid the ambiguous truth value error
            stop_atr_mult = 1.0
            tp_atr_mult = 2.0

        # Long entries
        entry_long = (signals == 1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        # Short entries
        entry_short = (signals == -1.0)
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals