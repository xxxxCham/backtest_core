from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(indicators):
            # Indicators available are ['atr']

            # LONG intent
            long_signal = np.zeros_like(indicators['atr'], dtype=np.float64)
            close_price = df['close'].copy()  # Assume 'close' is the dataframe column for closing price
            ema10 = ta.EMA(close_price, timeperiod=10).values  # Calculate EMA of 'close_price' with a period of 10

            rsi = ta.RSI(close_price, timeperiod=14)  # Calculate RSI for 'close_price' with a period of 14
            i = np.where(rsi > 50)[0]  # Find index where RSI > 50

            lower_band = ta.BollingerBandsLowerBand(close_price, timeperiod=2).values[i][-1]  # Get the last value of Bollinger Bands for 'close_price' in uptrend

            long_condition = close_price[-1] > ema10 and rsi[-1] > 50 and lower_band < ema10[-1]  # Check if conditions are met to go LONG at current bar end time (last element of 'close_price', 'rsi' and 'lower_band')
            long_signal[i[long_condition]] = 1.0  # Set the corresponding signals for conditions that are met

            upper_band = ta.BollingerBandsUpperBand(close_price, timeperiod=2).values[-1][-1]  # Get the last value of Bollinger Bands for 'close_price' in downtrend

            short_condition = close_price[-1] < ema30 and rsi[-1] < 50 and upper_band > ema30[-1]  # Check if conditions are met to go SHORT at current bar end time (last element of 'close_price', 'rsi' and 'upper_band')
            short_signal = -1.0 * np.where(short_condition, 1.0, 0.0)  # Set the corresponding signals for conditions that are met (-1.0 if true, 0.0 otherwise)

            signals[i[:-1]] = long_signal + short_signal  # Assign signals to all bars except current bar end time (last element of 'long_condition' and 'short_condition')

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
