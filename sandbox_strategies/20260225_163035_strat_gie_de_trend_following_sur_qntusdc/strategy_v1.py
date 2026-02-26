from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_ichimoku_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'ichimoku', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 3.8,
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
                max_val=5.0,
                default=3.8,
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

        # Extract indicator arrays
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])

        ich = indicators['ichimoku']
        tenkan = np.nan_to_num(ich["tenkan"])
        kijun = np.nan_to_num(ich["kijun"])

        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Cross detection helpers
        prev_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_plus[0] = np.nan
        prev_minus[0] = np.nan
        cross_pos_below_neg = (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus']) & (prev_plus >= prev_minus)
        cross_neg_below_pos = (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus']) & (prev_minus <= prev_plus)

        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_close_below_kijun = (close < kijun) & (prev_close >= kijun)
        cross_close_above_kijun = (close > kijun) & (prev_close <= kijun)

        # Entry conditions
        long_mask = (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (close > kijun) & (close > tenkan)
        short_mask = (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus']) & (close < kijun) & (close < tenkan)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.7)
        tp_atr_mult = params.get("tp_atr_mult", 3.8)

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
