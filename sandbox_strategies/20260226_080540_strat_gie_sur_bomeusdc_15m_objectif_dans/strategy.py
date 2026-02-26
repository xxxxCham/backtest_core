from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Strategy on BOMEUSDC 15m')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'stochastic', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fees': 10.0,
         'leverage': 2,
         'slippage': 5.0,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 0.67,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'fees': ParameterSpec(
                name='fees',
                min_val=5.0,
                max_val=40.0,
                default=10.0,
                param_type='float',
                step=0.1,
            ),
            'slippage': ParameterSpec(
                name='slippage',
                min_val=1.0,
                max_val=20.0,
                default=5.0,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=0.67,
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
        def generate_signals(df):
            signals = pd.Series(np.zeros(len(df)), index=df.index)

            for i, row in df[params['warmup']:].iterrows():
                # calculate bollinger bands and stochastics for the current bar
                close_price = row["close"]
                upper, middle, lower = ta.BBANDS(row["close"], timeperiod=20)
                slowk, slowd = ta.STOCH(row["high"].values, row["low"].values, row["close"].values, fastk_period=14)

                # set the buy condition based on bollinger bands and stochastic oscillator conditions
                if close_price > upper[-1] and middle[-1] < slowk[-1] and slowd[-1] > 85:
                    signals[i] = 1.0   #long position

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
