from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_adx_35')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 18,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.75,
         'supertrend_multiplier': 2.5,
         'tp_atr_mult': 3.0,
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
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=18,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
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

        # unwrap indicators
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr_vals = np.nan_to_num(indicators['atr'])
        close_vals = df["close"].values

        # Entry logic: strong trend + ADX > 35
        long_mask = (st_dir == 1) & (adx_val > 35)
        short_mask = (st_dir == -1) & (adx_val > 35)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP levels for entries
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.75))
        tp_mult = float(params.get("tp_atr_mult", 3.0))

        # Long entries
        long_entry = signals == 1.0
        df.loc[long_entry, "bb_stop_long"] = close_vals[long_entry] - stop_mult * atr_vals[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close_vals[long_entry] + tp_mult * atr_vals[long_entry]

        # Short entries
        short_entry = signals == -1.0
        df.loc[short_entry, "bb_stop_short"] = close_vals[short_entry] + stop_mult * atr_vals[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close_vals[short_entry] - tp_mult * atr_vals[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
