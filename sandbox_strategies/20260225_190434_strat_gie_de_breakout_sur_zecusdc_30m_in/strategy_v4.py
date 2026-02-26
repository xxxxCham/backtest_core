from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='zecusdc_30m_breakout_ichimoku_donchian_bollinger_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'donchian', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'donchian_period': 20,
         'ichimoku_period1': 9,
         'ichimoku_period2': 26,
         'ichimoku_period3': 52,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.5,
         'trail_atr_mult': 1.7,
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
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
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
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1,
                max_val=3,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=5.0,
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'trail_atr_mult': ParameterSpec(
                name='trail_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.7,
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

        # Bollinger
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])

        # Donchian
        dc = indicators['donchian']
        upper_dc = np.nan_to_num(dc["upper"])
        lower_dc = np.nan_to_num(dc["lower"])
        middle_dc = np.nan_to_num(dc["middle"])

        # Ichimoku
        ich = indicators['ichimoku']
        tenkan = np.nan_to_num(ich["tenkan"])
        kijun = np.nan_to_num(ich["kijun"])
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        cloud_upper = np.maximum(senkou_a, senkou_b)
        cloud_lower = np.minimum(senkou_a, senkou_b)

        # ATR
        atr = np.nan_to_num(indicators['atr'])

        # Long entry conditions
        long_cond = (
            (close > upper_dc)
            & (close > cloud_upper)
            & (tenkan > kijun)
            & (close > upper_bb)
        )
        long_mask = long_cond

        # Short entry conditions
        short_cond = (
            (close < lower_dc)
            & (close < cloud_lower)
            & (tenkan < kijun)
            & (close < lower_bb)
        )
        short_mask = short_cond

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long = (close < middle_dc)
        exit_short = (close > middle_dc)

        # Apply exits by resetting signals to 0.0 where exit conditions met after entry
        # We need to track positions; for simplicity, we clear signals on exit bars
        # but keep existing signals for earlier bars
        # Note: this simplistic approach may not fully handle position tracking
        signals[(signals == 1.0) & exit_long] = 0.0
        signals[(signals == -1.0) & exit_short] = 0.0

        # ATR-based SL/TP columns for long entries
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.5)

        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
