from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase_Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ATRmoving average true range, 2 for example.': 3,
         'BOLLER.std_dev': 2.5,
         'Leverage': 1,
         'RSI.period': 9,
         'RSImoving average crossover period in days, 20 for example.': 'Period of moving '
                                                                        'averages such as EMA '
                                                                        'and SMA',
         'change_type': 'both',
         'entry_short_logic': '',
         'exit_logic': '',
         'hypothesis': 'Using the family robust indicators such as RSI and Bollinger bands to '
                       'create a trading strategy with EMA and ATR moving averages',
         'leverage': 1,
         'parameter_specs': {'ATRand moving average true range, 3 for example.': 4,
                             'BOLLER.std_dev': 2.5,
                             'Leverage': 1,
                             'RSI.period': 9,
                             'RSImoving average crossover period in days, 20 for example.': 'Period '
                                                                                            'of '
                                                                                            'moving '
                                                                                            'averages '
                                                                                            'such '
                                                                                            'as '
                                                                                            'EMA '
                                                                                            'and '
                                                                                            'SMA'},
         'required': True,
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
        # Initialize booleans for long and short positions
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Implement explicit LONG / SHORT logic
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
