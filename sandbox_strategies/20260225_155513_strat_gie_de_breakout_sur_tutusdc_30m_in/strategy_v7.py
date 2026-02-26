from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_adx_breakout_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'kijun_period': 26,
         'leverage': 1,
         'senkou_span_b_period': 52,
         'stop_atr_mult': 2.0,
         'tenkan_period': 9,
         'tp_atr_mult': 3.5,
         'warmup': 50}

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
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'tenkan_period': ParameterSpec(
                name='tenkan_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'kijun_period': ParameterSpec(
                name='kijun_period',
                min_val=10,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'senkou_span_b_period': ParameterSpec(
                name='senkou_span_b_period',
                min_val=20,
                max_val=100,
                default=52,
                param_type='int',
                step=1,
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
                max_val=6.0,
                default=3.5,
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

        # Extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        # Cloud boundaries
        cloud_upper = np.maximum(senkou_a, senkou_b)
        cloud_lower = np.minimum(senkou_a, senkou_b)

        # Helper for cross detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_cloud_upper = np.roll(cloud_upper, 1)
        prev_cloud_upper[0] = np.nan
        prev_cloud_lower = np.roll(cloud_lower, 1)
        prev_cloud_lower[0] = np.nan
        cross_up = (close > cloud_upper) & (prev_close <= prev_cloud_upper)
        cross_down = (close < cloud_lower) & (prev_close >= prev_cloud_lower)

        # ADX increasing
        prev_adx = np.roll(adx_val, 1)
        prev_adx[0] = np.nan
        adx_increasing = (adx_val > prev_adx)

        # Entry conditions
        long_mask = cross_up & (adx_val > 25.0) & adx_increasing
        short_mask = cross_down & (adx_val > 25.0) & adx_increasing

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.5))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
