from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='BreakoutVolatility')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        def generate_signals(indicators):
            # Create empty Series to store signals
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Iterate over each bar in the DataFrame
            for i, row in df.iterrows():
                price = row['price']

                # Get Bollinger Bands and ATR values
                bollinger_upperband = indicators['bollinger']['upper'][i]
                bollinger_middleband = indicators['bollinger']['middle'][i]
                bollinger_lowerband = indicators['bollinger']['lower'][i]
                atr_value = indicators['atr'][i][0]

                # Calculate ATR-based stop loss and take profit levels for long positions
                if atr_value > 0:
                    bb_stop_long, bb_tp_long = calculate_sl_tp(price, bollinger_middleband)
                else:
                    bb_stop_long, bb_tp_long = calculate_ll_tp(price, bollinger_middleband)

                # Calculate ATR-based stop loss and take profit levels for short positions
                if atr_value > 0:
                    bb_stop_short, bb_tp_short = calculate_sl_tp(-price, bollinger_middleband)
                else:
                    bb_stop_short, bb_tp_short = calculate_ll_tp(price, bollinger_middleband)

                # Determine signal based on ATR expanding and price closing below Bollinger Lower Band for LONG positions
                if atr_value > 0 and row['close'] < bollinger_lowerband:
                    signals[i] = -1.0

                # Determine signal based on ATR expanding and price crossing above Bollinger Upper Band for SHORT positions
                elif atr_value < 0 and row['close'] > bollinger_upperband:
                    signals[i] = 1.0

            return signals
        return signals
