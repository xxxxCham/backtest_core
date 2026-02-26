from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='OPUSDC Momentum Breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'risk_multiplier': 2.0,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_length': ParameterSpec(
                name='aroon_length',
                min_val=8,
                max_val=40,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_k_val': ParameterSpec(
                name='stochastic_k_val',
                min_val=1,
                max_val=9,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_d_val': ParameterSpec(
                name='stochastic_d_val',
                min_val=1,
                max_val=9,
                default=3,
                param_type='int',
                step=1,
            ),
            'risk_multiplier': ParameterSpec(
                name='risk_multiplier',
                min_val=2.0,
                max_val=4.0,
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
        # Set up indicators
        indicators = {
            'aroon': np.array([0, 2]),
            'stochastic': dict(percD=np.linspace(1, 50, 5)), # You may need to adjust this based on your needs
        }

        def generate_signals(df):
            signals = pd.Series(0., index=df.index)

            for i in df['close'].index:
                close = df['close'][i]

                # LONG intent
                if (np.aroonie(df, 'up')[0][0] > 2/10 and stoch_kelly(df)[0]['percent_d'] > k and atr[0][0] > thresh):
                    signals[i] = 1. # You may need to adjust this based on your conditions for long signal

                # SHORT intent
                elif (np.aroonie(df, 'down')[0][0] < -2/10 and stoch_kelly(df)[0]['percent_d'] <- k and atr[0][0] > thresh): 
                    signals[i] = -1. # You may need to adjust this based on your conditions for short signal

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
