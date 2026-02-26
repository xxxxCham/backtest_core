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
        def generate_signals(df, default_params={}, indicators=None, leverage=1):
            # Input checks. If no 'indicators' provided or empty dataframe input, return None.
            if not isinstance(df, pd.DataFrame) or df.empty:
                print('Input Error: Invalid data frame')
                return None

            # Get available indicators from input parameters and default settings.
            if not indicators:
                indicators = default_params
            for key in indicators.keys():
                if key not in ['rsi', 'ema', 'atr']:
                    print(f'Error: Unknown indicator {key}')
                    return None

            # Generate signals based on available indicators and momentum strategy.
            for i, row in df.iterrows():
                close = np.array([row['close']])  # Assuming close price is last column

                if 'rsi' in indicators:
                    rsi_val = indicators['rsi']['rsi_' + str(indicators['rsi'].keys())]  # assuming indicator keys are unique strings
        signals.iloc[:warmup] = 0.0
        return signals
