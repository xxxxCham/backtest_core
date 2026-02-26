from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_momentum_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'momentum', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'momentum_period': 14,
         'stop_atr_mult': 1.5,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 2.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=7,
                max_val=25,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=2,
                max_val=5,
                default=3,
                param_type='float',
                step=0.1,
            ),
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
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
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])
        st = indicators['supertrend']
        supertrend = np.nan_to_num(st["supertrend"])
        direction = np.nan_to_num(st["direction"])

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_supertrend = np.roll(supertrend, 1)
        prev_supertrend[0] = np.nan

        long_entry = (close > supertrend) & (prev_close <= prev_supertrend) & (momentum > 0)
        short_entry = (close < supertrend) & (prev_close >= prev_supertrend) & (momentum < 0)

        long_exit = (close < supertrend) & (prev_close >= prev_supertrend)
        short_exit = (close > supertrend) & (prev_close <= prev_supertrend)

        long_mask = long_entry
        short_mask = short_entry

        long_mask[long_exit] = False
        short_mask[short_exit] = False

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
