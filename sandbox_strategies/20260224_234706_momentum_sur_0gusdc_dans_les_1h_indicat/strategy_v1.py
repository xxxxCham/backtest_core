from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'stochastic']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vwap': 4,
         'warmup': 50,
         'x': 1.5,
         'y': 2.0,
         'z': 0.3}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'x': ParameterSpec(
                name='x',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'y': ParameterSpec(
                name='y',
                min_val=1.5,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'z': ParameterSpec(
                name='z',
                min_val=0.1,
                max_val=0.4,
                default=0.3,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=1.5,
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

            # Implement explicit LONG / SHORT / FLAT logic
            # Warmup protection
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            # Write SL/TP columns into df if using ATR-based risk management
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
