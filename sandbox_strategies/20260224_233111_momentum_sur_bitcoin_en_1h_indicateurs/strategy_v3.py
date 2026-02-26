from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Voltige_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 9, 'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
        def generate_signals(self, df):
            # Input should be a DataFrame with columns 'close', 'signal' and other necessary columns

            long_mask = self.check_rsi()  # Check RSI condition for going long
            short_mask = self.check_macrotrend()  # Check macrotrend condition for going short

            signals = pd.Series(0.0, index=df.index)

            # Iterate through each bar in the DataFrame and set signal based on conditions
            for i, row in df.iterrows():
                if long_mask[i]:
                    signals[i] = 1.0  # Go long
                elif short_mask[i]:
                    signals[i] = -1.0  # Go short

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
