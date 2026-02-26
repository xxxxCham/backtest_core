from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ltcusdc_1w_sma_aroon_trend_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'aroon_period': 14,
            'atr_period': 14,
            'leverage': 1,
            'sma_period': 50,
            'stop_atr_mult': 2.2,
            'tp_atr_mult': 6.6,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=6.6,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        warmup = int(params.get("warmup", 50))
        # Masks are defined for clarity; they are not used directly but keep the
        # structure expected by the strategy framework.
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        close = df["close"].values
        sma = indicators['sma']  # already a numpy array

        # Cross conditions
        cross_up = (close > sma) & (np.roll(close, 1) <= np.roll(sma, 1))
        cross_down = (close < sma) & (np.roll(close, 1) >= np.roll(sma, 1))

        # Aroon conditions
        aroon_up = indicators['aroon']["aroon_up"]
        aroon_down = indicators['aroon']["aroon_down"]

        # Long and short signal conditions
        long_cond = cross_up & (aroon_up > 50) & (aroon_down < 50)
        short_cond = cross_down & (aroon_down > 50) & (aroon_up < 50)

        # Assign signals
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals.iloc[:warmup] = 0.0

        return signals