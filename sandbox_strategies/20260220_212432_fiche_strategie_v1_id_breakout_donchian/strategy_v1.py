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
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Implement explicit LONG / SHORT / FLAT logic...

                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                signals.iloc[:self.params["warmup"]] = 0.0 # implement warmup protection here

                # Implement ATR-based stop-loss and take-profit logic...
                atr = indicators['atr']
                close = df['close'].values
                k_sl = params['leverage'] * (1 + self.params["warmup"]) / atr - 0.5*self.params["stop_p"] # calculate stop loss level based on parameters
                tp = params['leverage'] * (1 - close[-1]/(close[i-1]-k_sl))**2 - k_sl+ self.params["take_profit"]# calculate take profit level based on parameters

                signals[(df > df.rolling(window=self.params['ema_window'], min_periods = 1).mean()) & (signals == 0) & (~long_mask)] = -1.0 # long signal for breakout above EMA
                signals[((df < df.rolling(window=self.params['ema_window']*2).min().values ) | ((close > k_sl))| ~short_mask) & (signals == 0)] = +1# short signal for breakdown below EMA or hitting take profit level

                # return signals
        signals.iloc[:warmup] = 0.0
        return signals
