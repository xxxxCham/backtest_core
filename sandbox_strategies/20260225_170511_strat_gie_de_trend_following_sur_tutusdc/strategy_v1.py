from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='macd_ichimoku_atr_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'ichimoku', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ichimoku_base_period': 26,
         'ichimoku_conversion_period': 9,
         'ichimoku_span_b_period': 52,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=10,
                max_val=50,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=3,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ichimoku_conversion_period': ParameterSpec(
                name='ichimoku_conversion_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ichimoku_base_period': ParameterSpec(
                name='ichimoku_base_period',
                min_val=10,
                max_val=30,
                default=26,
                param_type='int',
                step=1,
            ),
            'ichimoku_span_b_period': ParameterSpec(
                name='ichimoku_span_b_period',
                min_val=20,
                max_val=60,
                default=52,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=2.9,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        ich_kijun = np.nan_to_num(indicators['ichimoku']["kijun"])

        long_mask = (macd_hist > 0) & (close > ich_kijun)
        short_mask = (macd_hist < 0) & (close < ich_kijun)

        # Cross detection helpers
        prev_macd = np.roll(macd_hist, 1)
        prev_macd[0] = np.nan
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_kijun = np.roll(ich_kijun, 1)
        prev_kijun[0] = np.nan

        cross_macd_down = (macd_hist < 0) & (prev_macd >= 0)
        cross_close_below_kijun = (close < ich_kijun) & (prev_close >= prev_kijun)

        cross_macd_up = (macd_hist > 0) & (prev_macd <= 0)
        cross_close_above_kijun = (close > ich_kijun) & (prev_close <= prev_kijun)

        exit_long_mask = cross_macd_down | cross_close_below_kijun
        exit_short_mask = cross_macd_up | cross_close_above_kijun

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.9))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
