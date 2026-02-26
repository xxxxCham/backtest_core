from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_vortex_atr_trend')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 2.3,
         'tp_atr_mult': 3.9,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.9,
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
        # Boolean masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        close = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators['atr'])
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        vx = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vx["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vx["vi_minus"])

        # Cross helper functions
        def cross_up(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x < y) & (px >= py)

        # Entry logic
        long_mask = (close > senkou_a) & (senkou_a > senkou_b) & (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus'])
        short_mask = (close < senkou_b) & (senkou_b > senkou_a) & (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus'])

        # Exit logic
        exit_long_mask = cross_down(close, senkou_a) | (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus'])
        exit_short_mask = cross_up(close, senkou_b) | (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus'])

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.9))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
