from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': '60 ',
         'rsi_oversold': '30',
         'rsi_period ': 7,
         'stop_atr_mult': '1.25',
         'tp_atr_mult': '2.0',
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period ': ParameterSpec(
                name='rsi_period ',
                min_val=5,
                max_val=50,
                default=7,
                param_type='int',
                step=1,
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
                default='1.25',
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default='2.0',
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
        class BuilderGeneratedStrategy(StrategyBase):
            def __init__(self):
                super().__init__(name="FICHE_STRATEGIE v1")

            @property
            def required_indicators(self) -> List[str]:
                return ["bollinger", "atr", "macd"]

            @property
            def default_params(self) -> Dict[str, Any]:
                return {"leverage": 1, "warmup": 50} # Always include leverage

            @property
            def parameter_specs(self) -> Dict[str, ParameterSpec]:
                return {} # No need for any additional parameters in this case.

            def generate_signals(
                self,
                df: pd.DataFrame,
                indicators: Dict[str, Any],
                params: Dict[str, Any]
            ) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Implement your logic here to generate the signals based on FICHE_STRATEGIE v1 strategy. 
                # Make sure you handle long/short entry and risk management as per the requirements.
        signals.iloc[:warmup] = 0.0
        return signals
