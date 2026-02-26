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
        return ['adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=6.0,
                default=3.0,
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

                # implement explicit LONG / SHORT logic
                warmup = int(params.get("warmup", 50))
                signals.iloc[:warmup] = 0.0

                long_mask = np.zeros(len(df), dtype=bool)
                short_mask = np.zeros(len(df), dtype=bool)

                # implement ATR-based risk management using SL and TP levels computed from indicators['atr']
                atr = np.nan_to_num(indicators['atr'])
                long_stop_levels = df[signals == 1.0].index + warmup - len(df) + 1
                short_stop_levels = df[signals == -1.0].index + warmup - len(df) + 1

                sl_level, tp_level = params["leverage"], None # assuming leverage is given as a float and not used here

                long_mask[long_stop_levels] = True
                short_mask[short_stop_levels] = True

                signals[long_mask] = 1.0
                signals[-short_mask] = -1.0

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
