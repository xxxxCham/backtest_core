from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_trend_filter_vortex')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_avg_period': 20,
         'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
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
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        vortex = indicators['vortex']
        vortex_osc = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        # Precompute previous values for crossovers
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        prev_vortex_osc = np.roll(vortex_osc, 1)
        prev_close[0] = np.nan
        prev_ema[0] = np.nan
        prev_vortex_osc[0] = np.nan
        # Entry conditions
        # Long entry: close crosses above ema, close < lower bollinger, vortex rising from local min
        long_entry_cross = (close > ema) & (prev_close <= prev_ema)
        long_entry_below_bb = close < indicators['bollinger']['lower']
        long_entry_vortex_rising = vortex_osc > prev_vortex_osc
        long_mask = long_entry_cross & long_entry_below_bb & long_entry_vortex_rising
        # Short entry: close crosses below ema, close > upper bollinger, vortex rising from local min
        short_entry_cross = (close < ema) & (prev_close >= prev_ema)
        short_entry_above_bb = close > indicators['bollinger']['upper']
        short_entry_vortex_rising = vortex_osc > prev_vortex_osc
        short_mask = short_entry_cross & short_entry_above_bb & short_entry_vortex_rising
        # Exit conditions
        # Precompute previous bollinger values for exit conditions
        prev_bb_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_bb_upper[0] = np.nan
        prev_bb_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_bb_lower[0] = np.nan
        # Exit long: close crosses above upper bollinger or vortex decreasing
        long_exit_cross_bb = (close > indicators['bollinger']['upper']) & (prev_close <= prev_bb_upper)
        long_exit_vortex_falling = vortex_osc < prev_vortex_osc
        long_exit_mask = long_exit_cross_bb | long_exit_vortex_falling
        # Exit short: close crosses below lower bollinger or vortex decreasing
        short_exit_cross_bb = (close < indicators['bollinger']['lower']) & (prev_close >= prev_bb_lower)
        short_exit_vortex_falling = vortex_osc < prev_vortex_osc
        short_exit_mask = short_exit_cross_bb | short_exit_vortex_falling
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Compute dynamic stop loss based on ATR
        avg_atr = np.nanmean(atr)
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        if avg_atr > 0:
            # Adjust stop based on volatility
            if atr[-1] > avg_atr:
                stop_atr_mult = 1.0  # Tighter stop
            else:
                stop_atr_mult = 2.0  # Wider stop
        # Apply SL/TP for long entries
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        # Apply SL/TP for short entries
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals