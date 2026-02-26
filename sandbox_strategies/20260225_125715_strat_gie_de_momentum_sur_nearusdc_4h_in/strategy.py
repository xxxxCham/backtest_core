from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='near_usdc_roc_macd_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'macd_fast_period': 12,
            'macd_signal_period': 9,
            'macd_slow_period': 26,
            'roc_period': 14,
            'stop_atr_mult': 1.4,
            'tp_atr_mult': 2.9,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.9,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Initialise output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Retrieve required indicator arrays
        roc = indicators['roc']                     # plain array
        macd_hist = indicators['macd']['histogram']  # dict sub‑key

        # Previous ROC for acceleration check
        prev_roc = np.roll(roc, 1)

        # Build entry masks
        mask_long = (roc > 0) & (roc > prev_roc) & (macd_hist > 0)
        mask_short = (roc < 0) & (roc < prev_roc) & (macd_hist < 0)

        # Assign signals
        signals[mask_long] = 1.0
        signals[mask_short] = -1.0

        # Zero out warm‑up period
        signals[:warmup] = 0.0

        return signals