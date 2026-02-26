from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'Leverage': 1,
         'RSIPeriod': 14,
         'StopATRMult': 1.5,
         'TPATMult': 3.0,
         'WarmupPeriod': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'RSIPeriod': ParameterSpec(
                name='RSIPeriod',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'StopATRMult': ParameterSpec(
                name='StopATRMult',
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
            signals = pd.Series(0.0, index=df.index)  # initialize the signal series with zeros

            n = len(df)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            close = df['close'].values   # assuming 'close' is the column name for closing prices

            rsi_period = self.required_indicators[0]  # retrieve RSI period from default parameters
            bollinger_window = self.required_indicators[1]  # retrieve Bollinger Band window from default parameters
            atr_factor = params.get('atr', 1.)   # retrieve ATR factor from parameters or use the default value if not provided

            rsi_values = pd.Series(pd.StatsModulo.rolling(close, rsi_period).mean(), name='rsi') - 50  # compute RSI values
            upper, middle, lower = pd.StatsModulo.rolling(close, bollinger_window).bollinger()  # compute Bollinger Bands

            sl_level = close + atr_factor * rsi_values.std()   # calculate stop loss level using ATR
            tp_level = close - atr_factor * rsi_values.std()   # calculate take profit level using ATR

            long_mask[rsi_values < 30] = True   # set long signals when RSI is below 30 (oversold)
            short_mask[(rsi_values > 70) | (close.diff() < 0)] = True   # set short signals when RSI is above 70 (overbought) or price is decreasing

            signals[long_mask] = 1.0   # assign a value of 1 to long positions
            signals[short_mask] = -1.0   # assign a value of -1 to short positions

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
