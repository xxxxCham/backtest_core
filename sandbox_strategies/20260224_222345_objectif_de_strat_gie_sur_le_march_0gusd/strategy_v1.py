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
        return ['macd', 'ema', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_stddev': 2.0,
         'ema_length': 26,
         'leverage': 2,
         'macd_fast_length': 12,
         'macd_slow_length': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_length': ParameterSpec(
                name='ema_length',
                min_val=5,
                max_val=30,
                default=26,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'macd_fast_length': ParameterSpec(
                name='macd_fast_length',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_length': ParameterSpec(
                name='macd_slow_length',
                min_val=5,
                max_val=30,
                default=26,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(self, data):
            # Define inputs and outputs
            self.buy_price = symbol('close').latest - symbol('open')
            self.sell_price = symbol('close').latest * 1.25 - symbol('open')

            # Initialize signals as zeros (0)
            signals = pd.Series(np.zeros(len(data), dtype=float))

            for i, row in data.iterrows():
                close_price = float(row['close'])

                if self._should_buy(i):  # Use your logic to determine when to buy here
                    signals[i] = 1.0

                elif self._should_sell(i):  # Use your logic to determine when to sell here
                    signals[i] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
