from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'donchian', 'atr']

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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Define Bollinger Bands parameters
            window_size = params['bollinger']['window'] if 'bollinger' in params else 20
            num_stddevs = params['bollinger'].get('num_stddevs', 2)

            # Compute upper and lower BB bands
            df['upper_bb'], _ = pd.Series(df['close']).rolling(window=window_size, min_periods=1).agg([np.mean, np.std])[['mean']].values[0] + num_stddevs * np.array([-1, 1])
            df['lower_bb'], _ = pd.Series(df['close']).rolling(window=window_size, min_periods=1).agg([np.mean, np.std])[['mean']].values[0] - num_stddevs * np.array([-1, 1])

            # Compute the z-score for each data point
            df['z_score'], _ = pd.Series(df['close']).rolling(window=window_size, min_periods=1).agg([np.mean, np.std]).values[0] - params.get('bollinger', {}).get('shift', 2)

            # Compute the long/short signals based on Bollinger Bands and z-score
            signals[(df['close'] > df['upper_bb']) & (params['direction'].lower() == 'long')] = 1.0
            signals[((df['close'] < df['lower_bb']) | ((df['z_score'] >= num_stddevs) & params['direction'].lower() == 'short'))] = -1.0

            # Apply the signal to the original DataFrame
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
