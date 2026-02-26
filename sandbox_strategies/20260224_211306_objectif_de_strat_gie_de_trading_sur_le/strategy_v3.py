from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='My_Unique_Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'donchian', 'momentum']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'Leverage': 'Auto',
         'fees': 0.289,
         'leverage': 1,
         'rsi_period': 14,
         'slippage': 0.2,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volumefilter_fast': 8,
         'volumefilter_slow': 36,
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
        def generate_signals(indicators):
            # Aroon is above 80 and Momentum in fast period crosses over its slow period AND Donchian middle band is crossed upward by close price.
            long = (indicators['aroon'] > 80) & \
                   ((indicators['momentum'] - indicators['momentum']) > 0) & \
                    (indicators['donchian']['upper'][1] < close)

            # Momentum in slow period crosses under its fast period and Aroon is below 20 AND Donchian upper band is crossed downward by close price.
            short = ((indicators['momentum'] - indicators['momentum']) > 0) & \
                     (indicators['aroon'] < 20) & \
                     (indicators['donchian']['upper'][1] > close)

            # Long signals
            signals_long = long | ((indicators['donchian']['middle'] >= indicators['donchian']['lower']) & (close >= indicators['donchian']['upper']))

            # Short signals
            signals_short = short | ((indicators['donchian']['middle'] <= indicators['donchian']['upper']) & (close < indicators['donchian']['lower']))

            # Return the combined signals and update dataframe with them.
        signals.iloc[:warmup] = 0.0
        return signals
