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
        def generate_signals(df: pd.DataFrame) -> pd.Series:
            # Generate buy and sell signals using TA-Lib's BBANDS and RSI indicators

            bb = BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdown=2)[-1]
            rsi_val = RSI(df['close']).RSV[-1]

            if bb[3] > bb[1]:  # Upper BB is greater than the middle one (Bullish)
                buy_signal = True
            elif bb[2] < bb[0]:  # Middle BB is less than the lowest one (Bearish)
                sell_signal = True

            if rsi_val > 80:   # Overbought area
                sell_signal = True
            elif rsi_val < 20:  # Oversold area
                buy_signal = True

            signals = pd.Series([buy_signal, sell_signal], index=df.index)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
