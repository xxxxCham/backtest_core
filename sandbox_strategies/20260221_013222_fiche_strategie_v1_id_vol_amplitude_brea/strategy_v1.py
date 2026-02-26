from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='<Your Strategy Name>')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

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
        # Initialize signals DataFrame with zeros and write default parameters
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Iterate over each bar in the input DataFrame
        for i, row in df.iterrows():
            # Get indicator values for the current bar
            indicators_values = [row[indicator] for indicator in ['atr', 'bollinger', 'donchian', 'adx', 'supertrend', 'stochastic'] if indicator in row and isinstance(row[indicator], np.ndarray)]

            # Check conditions to enter long position (e.g., using AND operator)
            if all([val > val_thresh for val, val_thresh in zip(*indicators_values)]):
                signals[i] = 1.0

            # Check conditions to enter short position (e.g., using OR operator)
            elif any([val < val_thresh for val, val_thresh in zip(*indicators_values)]):
                signals[i] = -1.0

            else:
                signals[i] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
