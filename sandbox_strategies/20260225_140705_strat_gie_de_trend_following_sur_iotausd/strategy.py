from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_vortex_atr_trend_refined')

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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=50,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        vx = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vx["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vx["vi_minus"])

        atr = np.nan_to_num(indicators['atr'])

        # Long entry conditions
        long_mask = (
            (close > senkou_a)
            & (senkou_a > senkou_b)
            & (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus'])
            & ((indicators['vortex']['vi_plus'] - indicators['vortex']['vi_minus']) > 0.05)
        )

        # Short entry conditions
        short_mask = (
            (close < senkou_b)
            & (senkou_b > senkou_a)
            & (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus'])
            & ((indicators['vortex']['vi_minus'] - indicators['vortex']['vi_plus']) > 0.05)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare cross detection for exits
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_senkou_a = np.roll(senkou_a, 1)
        prev_senkou_a[0] = np.nan
        prev_senkou_b = np.roll(senkou_b, 1)
        prev_senkou_b[0] = np.nan

        cross_down_close_senkou_a = (close < senkou_a) & (prev_close >= prev_senkou_a)
        cross_up_close_senkou_b = (close > senkou_b) & (prev_close <= prev_senkou_b)

        # Exit conditions
        long_exit_mask = cross_down_close_senkou_a | (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus'])
        short_exit_mask = cross_up_close_senkou_b | (indicators['vortex']['vi_minus'] < indicators['vortex']['vi_plus'])

        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # ATR-based SL/TP levels
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
