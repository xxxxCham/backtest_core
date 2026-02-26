from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
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
                default=1.5,
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
        def generate_signals(indicators):
            df = pd.DataFrame() # replace this line by your dataframe data structure

            for index, row in df.iterrows():
                close = np.array([row['close']])  # replace this line with the correct column name and array conversion if needed

                long_intent = ((close - indicators['bollinger']["upper"]) > 0) & \
                               (np.any(close >= row[indicators['rsi']]["rsi"] > 50)) & \
                               (row[indicators['adx']]["adx"] > 25)   # replace this line with the correct indicator value comparison

                short_intent = ((close - indicators['bollinger']["lower"]) < 0) & \
                                (np.any(close <= row[indicators['rsi']]["rsi"] < 50)) & \
                                (row[indicators['adx']]["adx"] > 25)   # replace this line with the correct indicator value comparison

                if long_intent or short_intent:
                    signals.loc[index] = 1.0   # replace this line by your desired signal assignment logic

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
