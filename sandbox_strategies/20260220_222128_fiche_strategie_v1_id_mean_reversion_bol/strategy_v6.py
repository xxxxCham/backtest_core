from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='MyStrategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'rsi']

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
        def generate_signals(self, df: pd.DataFrame, indicators: List[str]) -> pd.Series:  # Here we declare that our function should take in a dataframe of data and a list of required indicators as parameters 
            signals = []

            for i, v in enumerate([sma, rsi]):  # iterate over the list of required indicators 'sma' and 'rsi'. Here we assume that both inputs are pandas DataFrame.
                long_mask = (df[v].rolling(window=2).mean() > df[v].rolling(window=2).mean().shift()) & \
                            (df[v].rolling(window=2).std() < df[v].rolling(window=2).mean().shift()) # check for bullish crossover
                signals.append((np.where(long_mask, 1 ,0)) )

            return pd.Series(signals)
        signals.iloc[:warmup] = 0.0
        return signals
