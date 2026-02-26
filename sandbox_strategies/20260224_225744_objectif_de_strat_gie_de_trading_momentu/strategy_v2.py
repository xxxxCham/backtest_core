from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ADXR Bollinger Breakout with ATR Risk Management')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adxr_period': 14,
         'bollinger_stddev': 2,
         'leverage': 2,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adxr_period': ParameterSpec(
                name='adxr_period',
                min_val=7,
                max_val=40,
                default=14,
                param_type='int',
                step=1,
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
                default=2,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3,
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
        def generate_signals(self, df, indicators, params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # implement explicit LONG / SHORT / FLAT logic
            long_mask = np.zeros(len(df), dtype=bool)
            short_mask = np.zeros(len(df), dtype=bool)

            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            # Write SL/TP columns into df if using ATR-based risk management
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
