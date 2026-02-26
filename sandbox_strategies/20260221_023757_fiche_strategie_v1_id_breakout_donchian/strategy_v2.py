from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_overbought': 80,
         'rsi_oversold': 20,
         'rsi_period': 89,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 40}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        # Implement your logic for generating signals here
        n_bars = len(df)
        long_mask = np.zeros(n_bars, dtype=bool)
        short_mask = np.zeros(n_bars, dtype=bool)

        if params["leverage"] == 1: # include leverage protection for initial testing
            signals.iloc[:warmup] = 0.0

        # Implement your logic for generating signals here using the "donchian" and "bollinger" indicators, as well as other necessary conditions
        signals.iloc[:warmup] = 0.0
        return signals
