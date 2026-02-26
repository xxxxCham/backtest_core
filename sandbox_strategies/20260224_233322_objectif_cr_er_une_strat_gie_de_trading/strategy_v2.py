from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Proposal')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'bollinger', 'donchian', 'keltner', 'macd', 'momentum', 'obv', 'onchain_smoothing', 'pi_cycle', 'pivot_points', 'psar', 'roc', 'standard_deviation', 'stoch_rsi', 'volume_oscillator', 'vwap', 'williams_r']

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
        def generate_signals(df, default_params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # LONG signal condition - close > ma && momentum > 0.75 && obv > psar.atr
            long_mask = df['close'] > df['ma'].rolling(window=14).mean() & \
                        (df[momentum] > 0.75) & \
                        (indicators['obv'] > df['psar.atr'])

            # SHORT signal condition - close < ma && momentum < -0.75 && obv < psar.atr
            short_mask = df['close'] < df['ma'].rolling(window=14).mean() & \
                          (df[momentum] < -0.75) & \
                          (indicators['obv'] < df['psar.atr'])

            # Assign long signals to the series, using 1.0 as value for both close and ma conditions
            signals[long_mask] = 1.0

            # Assign short signals to the series, using -1.0 as value for both close and ma conditions
            signals[short_mask] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
