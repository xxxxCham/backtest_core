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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df, long_intent, short_intent):
            indicators = {'rsi': df.rsi, 'ema': df.ema, 'atr': df.atr}  # Update with actual indicators here

            for indicator in indicators:
                arr = indicators[indicator]

                if indicator == 'bollinger' and any(k not in ['upper', 'middle', 'lower'] for k in long_intent):
                    raise ValueError('Invalid values for bollinger')

                elif indicator == 'adx':
                    adxs, indicators['adx']['plus_di'], indicators['adx']['minus_di'] = indicators["adx"]["adx"], adx[f'{indicators['adx']['plus_di']}_di'], adx[f'{indicators['adx']['minus_di']}_di']  # Update with actual indicators here

                elif indicator == 'supertrend':
                    supertrends, direction = indicators["supertrend"]["supertrend"], supertrend[direction]  # Update with actual indicators here

                elif indicator == 'stochastic' and f"{indicator}_k" not in df:
                    raise ValueError(f"'{indicator}' does not have a corresponding K value.")

                else:
                    arr = pd.Series([], index=df.index)  # Default empty signal array here for each indicator if it doesn't exist

            signals = signals + long_intent - short_intent  # Generate buy and sell signals based on long/short intent

            return signals[signals > 0]  # Return only positive signals as Buy signals
        return signals
