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
        return ['rsi']

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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            # Initialize long and short mask to False (not invested).
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # Implement your logic for entry signals generation here. For simplicity, let's assume that you have a function `entry_signal()` which returns True when the conditions are met.
            long_mask[indicators['rsi'] > 50] = self.entry_signal(df=df, indicators=indicators)

            # Implement your logic for exit signals generation here. For simplicity, let's assume that you have a function `exit_signal()` which returns True when the conditions are met and False otherwise.
            short_mask[self.exit_signal(df=df, indicators=indicators)] = True

            # Update signal based on long/short positions
            signals[(long_mask | short_mask) & ~entry_mask] = 1.0

            # Implement your logic for trailing stop loss here. For simplicity, let's assume that you have a function `trailing_stop()` which returns True when the conditions are met and False otherwise.
            signals[self.trailing_stop(df=df)] = self.entry_signal(df) - params['sl']

            # Implement your logic for taking profit here. For simplicity, let's assume that you have a function `take_profit()` which returns True when the conditions are met and False otherwise.
            signals[self.take_profit()] = self.entry_signal(df) + params['tp'] - signals[(long_mask | short_mask)].min()

            # Handle cases where no valid trades were entered or there was an invalid entry signal during the warmup period.
            if not long_mask.any():
                signals[:params["warmup"]] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
