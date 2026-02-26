from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_vortex_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_avg_period': 20,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

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

        # Compute previous values for crossovers
        prev_upper_bb = np.roll(upper_bb, 1)
        prev_lower_bb = np.roll(lower_bb, 1)
        prev_oscillator = np.roll(oscillator, 1)
        prev_vi_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vi_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_upper_bb[0] = np.nan
        prev_lower_bb[0] = np.nan
        prev_oscillator[0] = np.nan
        prev_vi_plus[0] = np.nan
        prev_vi_minus[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above lower BB and vortex oscillator is rising and > 0.5
        cross_above_lower = (close > prev_lower_bb) & (close <= lower_bb)
        vortex_rising = (oscillator > prev_oscillator)
        vortex_condition = (oscillator > 0.5)
        long_entry = cross_above_lower & vortex_rising & vortex_condition

        # Short entry: close crosses below upper BB and vortex oscillator is rising and > 0.5
        cross_below_upper = (close < prev_upper_bb) & (close >= upper_bb)
        short_entry = cross_below_upper & vortex_rising & vortex_condition

        # Exit conditions
        # Long exit: close crosses above upper BB or vortex oscillator is falling or < 0.5
        cross_above_upper = (close > prev_upper_bb) & (close <= upper_bb)
        vortex_falling = (oscillator < prev_oscillator)
        vortex_exit = (oscillator < 0.5)
        long_exit = cross_above_upper | vortex_falling | vortex_exit

        # Short exit: close crosses below lower BB or vortex oscillator is falling or < 0.5
        cross_below_lower = (close < prev_lower_bb) & (close >= lower_bb)
        short_exit = cross_below_lower | vortex_falling | vortex_exit

        # Set masks
        long_mask = long_entry
        short_mask = short_entry

        # Apply exits
        long_exit_mask = long_exit
        short_exit_mask = short_exit

        # For simplicity, assume that exit conditions override any existing positions
        # We only allow one position at a time
        # Update signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set SL/TP levels using ATR
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Determine volatility-based SL
        atr_avg = np.nanmean(atr)
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        # Long entries
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        # Short entries
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
