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
        return ['stochastic', 'atr', 'vortex', 'donchian']

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

        # Compute crossover conditions
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        cross_up = (k > d) & (prev_k <= prev_d)
        cross_down = (k < d) & (prev_k >= prev_d)

        # Entry conditions
        long_condition = cross_up & (k < 20) & (atr > np.nanmean(atr, axis=0)) & (vip > vim)
        short_condition = cross_down & (k > 70) & (atr > np.nanmean(atr, axis=0)) & (vim > vip)

        long_mask = long_condition
        short_mask = short_condition

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        lower = np.nan_to_num(dc["lower"])
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        exit_long = close < lower
        exit_short = close > upper
        exit_mask = exit_long | exit_short
        signals[exit_mask] = 0.0

        # Dynamic SL/TP based on ATR
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Dynamic multiplier based on volatility
        atr_max = np.nanmax(atr)
        atr_mean = np.nanmean(atr)
        vol_ratio = atr_mean / atr_max if atr_max != 0 else 0
        stop_mult = params["stop_atr_mult"]
        tp_mult = params["tp_atr_mult"]
        if vol_ratio < 0.5:
            stop_mult *= 0.75
            tp_mult *= 0.75
        elif vol_ratio > 0.8:
            stop_mult *= 1.25
            tp_mult *= 1.25

        # Write SL/TP levels only on entry bars
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals