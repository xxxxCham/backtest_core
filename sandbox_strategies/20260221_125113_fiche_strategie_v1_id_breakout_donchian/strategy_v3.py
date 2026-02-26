from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'bollinger', 'atr']

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
        def generate_signals(indicators, default_params):
            df = pd.DataFrame(**default_params)  # Assuming 'df' is your DataFrame and it has been passed in as a parameter to this function

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i, row in df.iterrows():
                long_intent = (row['close'] > indicators['bollinger']['upper']) & \
                              ((row['rsi'] > indicators['bollinger']['middle'])) & \
                              (row['atr'] > indicators['donchian']['middle'])

                short_intent = (row['close'] < indicators['bollinger']['lower']) & \
                               ((row['rsi'] < indicators['bollinger']['middle'])) & \
                               (row['atr'] < indicators['donchian']['middle'])

                if long_intent or short_intent:  # OR logic to ensure at least one of the conditions is met for a trade signal.
                    signals[i] = 1.0

                elif row['adx'] > 25 and (row['supertrend']['direction'] == 'up'):  # For example, you might use this line if adx or supertrend is your indicator of choice for a trade signal condition.
                    signals[i] = -1.0

                else:
                    signals[i] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
