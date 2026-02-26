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
        def generate_signals(df):
            # Set up default parameters
            default_params = {'leverage': 1}

            # Loop over each row in the DataFrame
            for i, bar in df.iterrows():
                long_intent = False
                short_intent = False

                # Determine momentum based on RSI and EMA signals
                rsi_value = np.NaN
                ema_value = np.NaN
                atr_value = np.NaN
                if 'rsi' in df: 
                    rsi_value = bar['rsi']
                if 'ema' in df: 
                    ema_value = bar['ema']
                if 'atr' in df: 
                    atr_value = bar['atr']

                # Calculate the momentum based on RSI and EMA values
                if np.abs(np.mean([rsi_value, ema_value])) < 30:
                    long_intent = True

                # Calculate the trend based on Bollinger Bands and Donchian Channels
                indicators['donchian']['upper'] = bar['close'].rolling(window=donchian_bandwidth).max()
                prev_upper = np.roll(indicators['donchian']['upper'], -1)

                if (bar['close'] > prev_upper[0]) and (long_intent or short_intent):  # Long intent
                    long_signal = 1.0

                elif (bar['close'] < indicators['donchian']['lower'][-2]) and (long_intent or short_intent):  # Short intent
                    short_signal = -1.0

                else:
                    long_signal, short_signal = 0.0, 0.0

                # Calculate the trend based on ATR-based stop loss/take profit levels
                if 'bb_stop' in df and 'bb_tp' in df:
                    bb_value = bar['close'] - bar[df['bb_stop'][i]]
                    atr_value = np.mean(bar['atr']) * 2 / len(bar)

                    long_stop, long_tp = bb_value + atr_value, bb_value + df['bb_tp'][i] + atr_value
                    short_stop, short_tp = bar[df['bb_stop'][i]] - bb_value - atr_value, bar[df['bb_tp'][i]] - bb_value - atr_value

                else:  # No stop loss/take profit levels provided
                    long_stop, long_tp = np.NaN, np.NaN
                    short_stop, short_tp = np.NaN, np.NaN

                if bar['close'] > long_stop and long_intent:  
                    signals[i] += 1.0

                elif (bar['close'] < short_stop) and short_intent:  # Short intent
                    signals[i] -= 1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
