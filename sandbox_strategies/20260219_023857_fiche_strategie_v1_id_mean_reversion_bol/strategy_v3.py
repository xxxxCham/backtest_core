from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR_stop_mult': ParameterSpec(
                name='ATR_stop_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'ATR_takeprofit_mult': ParameterSpec(
                name='ATR_takeprofit_mult',
                min_val=1.0,
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
        def generate_signals(indicators, bars):
            signals = pd.Series(0.0, index=bars.index, dtype=np.float64)

            # loop through each bar in the data
            for i, row in bars.iterrows():
                close = row['close']

                bollinger_upperband = indicators['bollinger']['upper'][i]
                bollinger_middleband = indicators['bollinger']['middle'][i]
                bollinger_lowerband = indicators['bollinger']['lower'][i]
                rsi = indicators['rsi'][i]
                atr = indicators['atr'][i]

                # check for LONG and SHORT signals based on close price crossing BBs or RSI value
                if (close > bollinger_upperband) & (indicators['RSI'] < 30):
                    signals[i] = 1.0

                elif (close < bollinger_lowerband) & (indicators['RSI'] > 70):
                    signals[i] = -1.0

            return signals
        return signals
