from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'donchian_return_period': 14,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_multiplier': ParameterSpec(
                name='atr_multiplier',
                min_val=0.5,
                max_val=3.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=8,
                max_val=26,
                default=8,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=12,
                max_val=26,
                default=26,
                param_type='int',
                step=1,
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
        def generate_signals(self):
            # Initialize variables here

            self._check_ticker()  # Check if stocks are AAPL or GOOG

            ticker = 'AAPL' if 'GOOG' not in self.stocks else 'GOOG'  

            signals = [0,1][np.random.randint(2)]  # Generate random signal for both stocks: buy/sell

            if signals[0] == 1 and signals[1] == 1:  # Both signals are active at the same time
                self._execute_buy_signals()   # Execute buy orders
            elif signals[0] == 1 or signals[1] == 1:  # Only one signal is active at a time
                if signals[0] == 1:  # First stock's signal is active, execute buy order for it
                    self._execute_buy_signals()

            elif signals[0] == -1 and signals[1] == -1:   # Both signals are inactive at the same time
                if np.random.randint(2) == 1:  # One stock's signal is active randomly, execute sell order for it
                    self._execute_sell_signals()
        signals.iloc[:warmup] = 0.0
        return signals
