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
        return ['ema', 'sma', 'macd', 'roc', 'aroon']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATR_Risk_multiplier': 1.0,
         'Aroon_Threshold_Down': 25,
         'Aroon_Threshold_Up': 75,
         'Bollinger_Threshold': 95,
         'Vwap_threshold': 80,
         'leverage': 1,
         'parameter_specs': {'ATR_Risk_multiplier': {'default': 1, 'max': 4.0, 'min': 1.0},
                             'Aroon_Threshold_Down': {'default': 25, 'max': -100, 'min': -80},
                             'Aroon_Threshold_Up': {'default': 75, 'max': '100', 'min': '20'},
                             'Bollinger_Threshold': {'default': 95, 'max': 95, 'min': 80},
                             'Vwap_threshold': {'default': 80, 'max': 95, 'min': 60}},
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

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
            n = len(df)

            # Implement explicit LONG / SHORT logic here
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            warmup = int(params.get("warmup", 50))

            # Write SL/TP columns into df if using ATR-based risk management
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
