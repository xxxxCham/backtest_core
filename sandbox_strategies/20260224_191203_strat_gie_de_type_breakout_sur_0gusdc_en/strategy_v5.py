from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_stochastic_breakout')

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
        donchian = indicators['donchian']
        indicators['donchian']['lower'] = np.nan_to_num(indicators['donchian']["lower"])
        indicators['donchian']['upper'] = np.nan_to_num(indicators['donchian']["upper"])
        # Compute previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        # Entry conditions
        # Long entry: k crosses above d, k < 20, atr > atr.mean(20), vip > vim
        long_cross_up = (k > d) & (prev_k <= prev_d)
        long_k_below_20 = k < 20
        atr_mean_20 = np.nanmean(atr)
        long_atr_filter = atr > atr_mean_20
        long_vortex_confirm = vip > vim
        long_entry = long_cross_up & long_k_below_20 & long_atr_filter & long_vortex_confirm
        long_mask = long_entry
        # Short entry: k crosses below d, k > 80, atr > atr.mean(20), vim > vip
        short_cross_down = (k < d) & (prev_k >= prev_d)
        short_k_above_80 = k > 80
        short_atr_filter = atr > atr_mean_20
        short_vortex_confirm = vim > vip
        short_entry = short_cross_down & short_k_above_80 & short_atr_filter & short_vortex_confirm
        short_mask = short_entry
        # Exit conditions: close crosses below indicators['donchian']['lower'] or above indicators['donchian']['upper']
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        exit_long = close < indicators['donchian']['lower']
        exit_short = close > indicators['donchian']['upper']
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP logic
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Dynamic SL based on ATR relative to recent max
        atr_window = params.get("atr_period", 14)
        atr_max = np.maximum.accumulate(atr)
        atr_ratio = atr / atr_max
        # Determine SL/TP multiplier based on volatility
        sl_mult = np.where(atr_ratio < 0.5, 1.5, 1.5)
        tp_mult = np.where(atr_ratio > 0.8, 3.0, 3.0)
        # Set SL/TP for long entries
        entry_long = (signals == 1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - sl_mult[entry_long] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult[entry_long] * atr[entry_long]
        # Set SL/TP for short entries
        entry_short = (signals == -1.0)
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + sl_mult[entry_short] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult[entry_short] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals