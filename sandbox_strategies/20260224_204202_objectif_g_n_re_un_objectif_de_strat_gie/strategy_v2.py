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
        # Generate signals based on indicators
        def generate_signals(df, default_params):
            # Get available indicators
            indicators = default_params['indicators']

            # Define long and short signals
            LONG = 'LONG'
            SHORT = 'SHORT'

            # Loop through each row in the dataframe
            for i, row in df.iterrows():
                close = np.array(row[['close', 'atr']])  # ATR-based stop loss and take profit levels

                if indicators['bollinger']['upper'][i] > close:
                    signals[i] = LONG  # Long signal

                    # Calculate take profit level using Bollinger Bands upper band
                    tp_level = row['close'] + bollinger.get('atr', [])[-1]*2
                elif indicators['bollinger']['middle'][i] > close:
                    signals[i] = LONG  # Long signal with tighter stop loss

                    # Calculate take profit level using Bollinger Bands middle band
                    tp_level = row['close'] + bollinger.get('atr', [])[-1]*2
                elif indicators['bollinger']['lower'][i] > close:
                    signals[i] = LONG  # Long signal with tighter stop loss and take profit level below previous low

                    # Calculate take profit level using Bollinger Bands lower band
                    tp_level = row['close'] + bollinger.get('atr', [])[-1]*2
                else:
                    signals[i] = SHORT  # Short signal with tight stop loss and target above recent high

                    # Define upper Donchian channel using previous day's close price as lower band
                    indicators['donchian']['lower'] = row['close'] - bollinger.get('donchian', [])[-1]*2

                    # Calculate take profit level using Donchian Breakout method with tight stop loss and target above recent high
        signals.iloc[:warmup] = 0.0
        return signals
