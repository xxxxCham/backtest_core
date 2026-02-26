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
        return ['ema', 'aroon', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATR Multiplier for SL': 1.9873648,
         'Leverage': 2.0,
         'SL at % IV': 0.997364804,
         'TP at BB high using ATR trailing stop loss': 1.987364804,
         'Warmup Period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'Warmup Period': ParameterSpec(
                name='Warmup Period',
                min_val=50,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'ATR Multiplier for SL': ParameterSpec(
                name='ATR Multiplier for SL',
                min_val=1.987364804,
                max_val=3,
                default=1.987364804,
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
        # Your logic here to generate signals based on the inputs and your strategy
        pass
        signals.iloc[:warmup] = 0.0
        return signals
