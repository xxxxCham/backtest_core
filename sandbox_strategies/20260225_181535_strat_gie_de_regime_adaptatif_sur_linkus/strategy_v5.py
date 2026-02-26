from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='regime_adaptive_keltner_30m_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'keltner', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.01,
         'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 4.8,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.001,
                max_val=0.02,
                default=0.01,
                param_type='float',
                step=0.001,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.8,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
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

        # Wrap indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        middle = np.nan_to_num(kelt["middle"])
        lower = np.nan_to_num(kelt["lower"])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])

        # Helper cross functions
        def cross_up(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Volatility regimes
        vol_high = atr > params.get("atr_threshold", 0.01)
        adx_strong = adx_arr > 25
        adx_weak = adx_arr < 20

        # Long entries
        long_vol_breakout = cross_up(close, upper) & vol_high & adx_strong
        long_mean_revert = cross_up(close, middle) & (~vol_high) & adx_weak
        long_mask = long_vol_breakout | long_mean_revert

        # Short entries
        short_vol_breakout = cross_down(close, lower) & vol_high & adx_strong
        short_mean_revert = cross_down(close, middle) & (~vol_high) & adx_weak
        short_mask = short_vol_breakout | short_mean_revert

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.7)
        tp_atr_mult = params.get("tp_atr_mult", 4.8)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
