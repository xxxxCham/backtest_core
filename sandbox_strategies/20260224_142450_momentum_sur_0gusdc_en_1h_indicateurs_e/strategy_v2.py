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
        return {'ema_fastperiod': 5,
         'leverage': 1,
         'obv_ignoreval': 0,
         'stochastic_slowd_period': 3,
         'stochastic_slowk_period': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fastperiod': ParameterSpec(
                name='ema_fastperiod',
                min_val=1,
                max_val=25,
                default=5,
                param_type='int',
                step=1,
            ),
            'obv_ignoreval': ParameterSpec(
                name='obv_ignoreval',
                min_val=-100,
                max_val=100,
                default=0,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                warmup = int(params.get("warmup", 50))
                signals.iloc[:warmup] = 0.0 # Setting the first few signals to zero initially

                # Write SL/TP levels into df if using ATR-based risk management
                sl_levels = pd.Series([], dtype=np.float64)
                tp_levels = pd.Series([], dtype=np.float64)

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
