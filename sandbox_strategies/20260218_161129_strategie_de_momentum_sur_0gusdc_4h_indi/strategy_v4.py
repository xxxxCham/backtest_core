from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_on_gusd_4h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr', 'rsi']

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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self):
            import numpy as np

            close = self.dataclose[0]  # assuming this datafeed provides a single column of 'close' values
            shift = int(np.floor(0.1 * len(close)))   # choose your own parameter for the lookback window size
            if not hasattr(self, 'signals'):
                self.signals = np.zeros(len(close), dtype=np.bool)  # initialize array with zeros of bool type

            # compute a running average for each close value over the past `lookback_window` days
            avg = np.cumsum(np.insert(close, 0, close[-1] - (len(close)-shift), axis=0))/float(len(close) + shift*2)

            # apply a threshold to the running average and create signal values above or below this value
            self.signals = avg > np.mean(avg) * 3   # change '3' parameter as needed for your specific use case
        return signals
