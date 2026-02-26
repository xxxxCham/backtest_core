from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'asynce_long_short_threshold': 'N/A',
         'asynce_long_short_vote_majority': 'N/A',
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'stop_loss_mult': 1.3,
         'take_profit_mult': 2.6,
         'tp_atr_mult': 3.0,
         'volatility_threshold': '5',
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'volatility_threshold': ParameterSpec(
                name='volatility_threshold',
                min_val=0,
                max_val=10,
                default=5,
                param_type='int',
                step=1,
            ),
            'stop_loss_mult': ParameterSpec(
                name='stop_loss_mult',
                min_val=1.2,
                max_val=4.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'take_profit_mult': ParameterSpec(
                name='take_profit_mult',
                min_val=2,
                max_val=8,
                default=2.6,
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
        def generate_signals(df, indicators, params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # implement explicit LONG / SHORT / FLAT logic
            # warmup protection
            warmup = int(params["warmup"])
            signals.iloc[:warmup] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
