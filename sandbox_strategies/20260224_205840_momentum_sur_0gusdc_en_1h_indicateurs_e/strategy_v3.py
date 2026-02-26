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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            n = len(df)
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Calculate RSI based on the provided indicators
            rsi_periods = 14  # Number of periods for RSI calculation (default is 14)
            rsi_values = self._calculate_rsi(df[df['close']], rsi_periods)

            # Calculate Bollinger Bands based on the provided indicators
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(df[df['close']])

            long_mask = np.where((rsi_values > 30) & (signals == 0))[0]
            short_mask = np.where((rsi_values < 70) & (signals != 1))[0]

            signals[(upper_band - middle_band).abs() < 2.5][long_mask] = 1.0
            signals[(middle_band - lower_band).abs() > 2.5][short_mask] = -1.0

            # Return the generated signals as a pandas Series
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
