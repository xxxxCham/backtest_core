from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGY')

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
        def generate_signals(df):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            n = len(df)

            # Set 'leverage' parameter to 1 if it is not provided in the default_params dictionary.
            # If you provide a value for leverage, ignore this condition.
            # Leverage should be set to 1 by default as per your requirements.
            # This line of code will be useful only when 'leverage' parameter is included in the parameters dictionary.

            if 'leverage' in params and params['leverage'] != 1:
                signals = np.where((df.high - df.close) > (2 * df.atr), 1, 0).astype(int)
            else:
                signals = np.where(((df.close - df.open) < -(params['leverage'] / params['lookback'])) & \
                                   ((df.high - df.low) > (2 * df.atr)), 1, 0).astype(int)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
