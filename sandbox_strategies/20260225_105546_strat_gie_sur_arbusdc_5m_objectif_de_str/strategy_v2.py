from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='CryptoTrendStrategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'bollinger']

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
        def generate_signals(df, default_params):
            # Calculate SMA and Bollinger Bands
            sma = df['close'].rolling(window=20).mean()
            upper, middle, lower = indicators['bollinger']['upper'], indicators['bollinger']['middle'], indicators['bollinger']['lower']

            # Define indicators to be calculated only once on start up
            if not hasattr(self, '_indicators'):
                self._indicators = {
                    'sma': sma.values, 
                    'bollinger': {'upper': upper, 'middle': middle, 'lower': lower}
                }

            # Calculate indicators and create signals based on conditions
            for i in df.index:

                # LONG intent - close > SMA(X) AND close < BOLLINGER(Y)
                if np.all([sma[i] >= sma_value, lower[i][0] <= df['close'][i] < upper[i][0]]):
                    signals[i] = 1.0

                # SHORT intent - close < SMA(X) AND close > BOLLINGER(Y)
                elif np.all([sma[i] <= sma_value, df['close'][i] >= lower[i][0], df['close'][i] <= upper[i][0]]):
                    signals[i] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
