from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_vortex_stochastic')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'atr', 'vortex', 'donchian', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=2,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])
        vortex = indicators['vortex']
        vip = np.nan_to_num(indicators['vortex']["vi_plus"])
        vim = np.nan_to_num(indicators['vortex']["vi_minus"])
        # Compute previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        # Crossover conditions
        cross_up = (k > d) & (prev_k <= prev_d)
        cross_down = (k < d) & (prev_k >= prev_d)
        # Entry conditions
        atr_mean = np.nanmean(atr)
        vol_filter = atr > atr_mean
        # Long entry: k crosses above d, k < 20, vol > mean, vortex confirms uptrend
        long_condition = cross_up & (k < 20) & vol_filter & (vip > vim)
        long_mask = long_condition
        # Short entry: k crosses below d, k > 80, vol > mean, vortex confirms downtrend
        short_condition = cross_down & (k > 80) & vol_filter & (vim > vip)
        short_mask = short_condition
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Exit conditions
        close = df["close"].values
        dc = indicators['donchian']
        dc_lower = np.nan_to_num(dc["lower"])
        dc_upper = np.nan_to_num(dc["upper"])
        kc = indicators['keltner']
        kc_upper = np.nan_to_num(kc["upper"])
        kc_lower = np.nan_to_num(kc["lower"])
        # Exit long if close crosses below donchian lower or close crosses above keltner upper
        exit_long = (close < dc_lower) | (close > kc_upper)
        # Exit short if close crosses above donchian upper or close crosses below keltner lower
        exit_short = (close > dc_upper) | (close < kc_lower)
        # Apply exit signals
        signals[exit_long & (signals == 1.0)] = 0.0
        signals[exit_short & (signals == -1.0)] = 0.0
        # Dynamic SL/TP based on ATR
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Volatility rotation logic
        max_atr = np.nanmax(atr)
        vol_ratio = atr / max_atr
        # Adjust SL/TP multipliers based on volatility
        stop_mult = np.where(vol_ratio < 0.5, params["stop_atr_mult"], 
                             np.where(vol_ratio > 0.8, params["stop_atr_mult"] * 2, params["stop_atr_mult"]))
        tp_mult = np.where(vol_ratio < 0.5, params["tp_atr_mult"], 
                           np.where(vol_ratio > 0.8, params["tp_atr_mult"] * 2, params["tp_atr_mult"]))
        # Compute SL/TP levels for long entries
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult[entry_long] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult[entry_long] * atr[entry_long]
        # Compute SL/TP levels for short entries
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult[entry_short] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult[entry_short] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals