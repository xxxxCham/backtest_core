from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ATUSDC Breakout with ATR and Volatility')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 5.0,
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
                min_val=2.0,
                max_val=3.0,
                default=2.0,
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
                default=5.0,
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
            df = pd.DataFrame(...) # Assuming a DataFrame of OHLC data is passed to the function here

            # Loop over each bar in the dataframe and check conditions
            for i, row in df.iterrows():
                close = row['close']  # Assuming 'close' column is available in `df`

                long_intent = (row['close'] > indicators['bollinger']['middle']) & \
                              (indicators['rsi'][i] > 70) & \
                              (indicators['adx'][i] < 25)

                short_intent = (row['close'] < indicators['bollinger']['lower']) & \
                               (indicators['rsi'][i] > 50) & \
                               (indicators['adx'][i] > 30)

                # Check if long or short intent is met, and set corresponding signal value
                signals[i] = np.where((long_intent | short_intent), -1.0, 0.0)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
