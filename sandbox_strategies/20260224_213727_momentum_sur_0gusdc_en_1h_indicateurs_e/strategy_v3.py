from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Calculate ATR values for the upper and lower Bollinger Bands
            atr_upper, atr_middle, atr_lower = self.__bb.get_atr([self.__close])

            # Generate signals based on whether close price is above or below EMA
            long_condition = (self.__close > self.__ema) & (self.__close >= self.__close[0] + 2 * atr_middle - self.__atr.get(self))
            short_condition = (self.__close < self.__ema) & (self.__close <= self.__close[-1] - 2 * atr_lower + self.__atr.get(self))

            # Place trades at the close price and exit when the stop loss is hit or a take profit target is reached
            for i, condition in enumerate([long_condition]*len(df)):
                if condition:
                    signals[i] = 1.0

                elif not long_condition[i]:
                    signals[i] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
