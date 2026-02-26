from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='xplusdc_regime_adaptive_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 30}

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
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.4,
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
        # Extract indicator arrays
        close = df['close'].values
        open_ = df['open'].values
        obv = indicators['obv']
        adx = indicators['adx']['adx']
        atr = indicators['atr']

        # Previous values (shifted by one)
        obv_prev = np.roll(obv, 1)
        atr_prev = np.roll(atr, 1)

        # Avoid using the first element as a valid comparison
        obv_prev[0] = np.nan
        atr_prev[0] = np.nan

        # Long and short entry masks
        long_mask = (
            (close > open_) &
            (obv > obv_prev) &
            (adx > 25) &
            (atr > atr_prev * 1.2)
        )

        short_mask = (
            (close < open_) &
            (obv < obv_prev) &
            (adx > 25) &
            (atr > atr_prev * 1.2)
        )

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
