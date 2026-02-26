from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=8.0,
                default=6.0,
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
        def generate_signals(indicators, intents):
            df = pd.DataFrame()  # Initialize an empty dataframe for storing data

            # Iterate through each bar in our DataFrame
            for i, row in df.iterrows():
                close = np.array([row['close']])  # Convert the 'Close' column to a numpy array

                # Check if we have more than one indicator available
                if len(indicators) > 1:
                    upper_band = indicators['donchian']['upper'].get(i, row['close'])  # Use the first Donchian band
                    prev_upper = np.roll(upper_band, -1)[0]  # Get previous Donchian band

                    if close > prev_upper and indicators['adx']["adx"]['adx'] > 20:
                        intents += 'AND'  # Add the AND intent for the next condition

                else:
                    superimposed_tr = indicators[long]['superimposed_tr'].get(i, row['close'])  # Get the first SuperTrend value

                    if close > donchian['donchian']['middle'] and indicators['adx']["adx"]['adx'] <= 15:
                        intents += 'AND'

                for intent in intents.split('&'):
                    intent = intent[0]  # Get the first character of our intent string

                    if close > donchian['donchian']['middle']:
                        signals[(i, 'long')] = 1.0  # Buy signal when we're at a new high

                    else:
                        signals[(i, 'long')] = -1.0  # Sell signal when the price breaks below the middle band

                for intent in intents.split('&'):
                    intent = intent[0]  # Get the first character of our intent string

                    if close > donchian['donchian']['middle']:
                        signals[(i, 'short')] = -1.0  # Sell signal when we're at a new low

                    else:
                        signals[(i, 'short')] = 1.0  # Buy signal when the price breaks above the middle band

                for intent in intents.split('&'):
                    intent = intent[0]  # Get the first character of our intent string

                    if close > donchian['donchian']['middle']:
                        signals[(i, 'long')] = 1.0  # Buy signal when we're at a new high

                    else:
                        signals[(i, 'short')] = -1.0  # Sell signal when the price breaks below the middle band

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
