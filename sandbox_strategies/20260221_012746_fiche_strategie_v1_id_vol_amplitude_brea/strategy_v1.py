from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vol_amplitude_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['amplitude_hunter', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'no_lookahead': True,
         'only_registry_indicators': True,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=4.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(indicators):
            # create a copy of df to avoid changes during iteration
            df = df.copy()

            for index, row in df.iterrows():
                price = row['close']

                # Apply indicators and conditions
                arr1 = row[list(indicators['amplitude_hunter'].keys())] 
                score = np.array([val if key == 'score' else None for key, val in markers.items()])
                close = np.roll(price, -1)   # TODO: Handle this properly with an indicator like ATR

                # Apply conditions
                long_signal = (arr1 > 0).all() and score[0] > 0.7 and price > arr1[0] 
                short_signal = ((arr1 < indicators['donchian']["middle"]).any() or close < indicators['donchian']["lower"]) \
                               and score[-1] < 0.35 and price < close  

                # Assign signals based on conditions
                if long_signal:
                    signals[index] = 1.0     # Buy signal
                elif short_signal:
                    signals[index] = -1.0    # Sell signal
                else:
                    signals[index] = 0.0   # No Signal

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
