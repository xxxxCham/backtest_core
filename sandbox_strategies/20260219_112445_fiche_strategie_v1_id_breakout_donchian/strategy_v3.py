from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'donchian']

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
        def generate_signals(
                self,
                df: pd.DataFrame,
                indicators: Dict[str, Any],
                params: Dict[str, Any]
            ) -> pd.Series:

                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Calculate Bollinger Bands using middle BB indicator
                upper_band = indicators['bollinger']['upper']
                lower_band = indicators['bollinger']['lower']
                close = df["close"]
                bb = pd.Series(((close - lower_band) * 2 + upper_band), index=df.index)

                # Calculate Donchian Channels using middle channel indicator and N days of data (N is a tunable parameter, default is 20)
                n_days = int(params['donchian']['n']) if 'donchian' in params else 20
                high = df["high"].rolling(window=n_days).max()
                low = df["low"].rolling(window=n_days).min()

                donchian_channels = pd.Series((high - low), index=df.index)

                # Use Bollinger Bands and Donchian Channels to generate trading signals
                long_mask = (signals == 0) & ((bb > upper_band) | (donchian_channels < lower_band))
                short_mask = (signals == 0) & ((bb < lower_band) | (donchian_channels > high))

                signals[long_mask] = 1.0 # Buy signal when price is above the upper Bollinger Band and below Donchian Channels
                signals[short_mask] = -1.0 # Sell signal when price is below the lower Bollinger Band and above Donchian Channels

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
