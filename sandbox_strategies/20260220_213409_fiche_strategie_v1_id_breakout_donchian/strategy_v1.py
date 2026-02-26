from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'no_lookahead': True,
         'only_registry_indicators': True,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
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
            # Convert input parameters to their expected types
            n = int(params['n'])  # n period Bollinger Bands
            k = float(params['k'])  # K factor for ATR calculation

            signals = np.zeros_like(df)  # Initialize signal series with zeros

            # Calculate upper and lower Bollinger Band and upper and lower ATR levels
            close = df['close'].values
            indicators['bollinger']['upper'], indicators['bollinger']['lower'] = self._get_bollinger_bands(n, k, close)
            atr_upper, atr_lower = self._calculate_atr(df['high'], df['low'], n, k)

            # Compute a long signal when close price crosses above the upper Bollinger Band and below the lower Bollinger Band
            mask1 = (close > indicators['bollinger']['lower'] - params["bollinger_width"]) & (close < indicators['bollinger']['upper'] + params["bollinger_width"])
            signals[mask1] = 1.0

            # Compute a short signal when close price crosses below the lower Bollinger Band and above the upper Bollinger Band
            mask2 = (close > indicators['bollinger']['upper'] - params["bollinger_width"]) & (close < indicators['bollinger']['lower'] + params["bollinger_width"])
            signals[mask2] = -1.0

            # Compute a long signal when close price crosses above the upper Bollinger Band and below the lower ATR level, or crosses below the lower Bollinger Band and above the upper ATR level
            mask3 = (close > atr_lower + params["atr_width"]) & (close < indicators['bollinger']['upper']) | (close > indicators['bollinger']['lower'] - params["atr_width"]) & (close < indicators['bollinger']['upper'])
            signals[mask3] = 1.0

            # Compute a short signal when close price crosses below the lower Bollinger Band and above the upper ATR level, or crosses above the upper Bollinger Band and below the lower ATR level
            mask4 = (close > atr_upper + params["atr_width"]) & (close < indicators['bollinger']['lower']) | (close > indicators['bollinger']['upper'] - params["atr_width"]) & (close < indicators['bollinger']['lower'])
            signals[mask4] = -1.0

            # Compute a long signal when close price crosses above the upper ATR level and below the lower Bollinger Band, or crosses below the lower ATR level and above the upper Bollinger Band
            mask5 = (close > atr_lower + params["atr_width"]) & (close < indicators['bollinger']['upper']) | (close > indicators['bollinger']['upper'] - params["atr_width"]) & (close < indicators['bollinger']['lower'])
            signals[mask5] = 1.0

            # Compute a short signal when close price crosses below the lower ATR level and above the upper Bollinger Band, or crosses above the upper ATR level and below the lower Bollinger Band
            mask6 = (close > atr_upper + params["atr_width"]) & (close < indicators['bollinger']['lower']) | (close > indicators['bollinger']['lower'] - params["atr_width"]) & (close < indicators['bollinger']['upper'])
            signals[mask6] = -1.0

            # Write signal to DataFrame and return it
            signals = signals
            return signals
        signals.iloc[:warmup] = 0.0
        return signals
