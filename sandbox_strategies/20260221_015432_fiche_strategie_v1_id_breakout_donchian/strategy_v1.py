from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
        # Indicators available in this method: ['rsi', 'ema', 'atr']
        indicators = {'rsi': None, 'ema': None, 'atr': None}

        # LONG intent: Entrée long si momentum haussier confirmé et risque contrôlé.
        def generate_long(df):
            # Select appropriate indicators and assign them to `indicators` dict
            for ind, val in {'rsi': 14, 'atr': 20}.items():
                if ind not in indicators:
                    signals[ind] = np.zeros((len(df), ))
                    signals[ind][:] = 0.0
                    indicators[ind] = df['close'].rolling(val).mean()
            # Write code to identify long entry based on momentum confirmed and risk controlled here

        # SHORT intent: Entrée short si momentum baissier confirmé et risque contrôlé.
        def generate_short(df):
            for ind, val in {'rsi': 14, 'atr': 20}.items():
                if ind not in indicators:
                    signals[ind] = np.zeros((len(df), ))
                    signals[ind][:] = 0.0
                    indicators[ind] = df['close'].rolling(val).mean()
            # Write code to identify short entry based on momentum confirmed and risk controlled here

        # In both generate_long & generate_short functions, write the logic for identifying a long or short trade based on the momentum condition.
        signals.iloc[:warmup] = 0.0
        return signals
