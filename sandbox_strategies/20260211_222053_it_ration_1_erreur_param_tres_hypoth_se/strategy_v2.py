from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_momentum_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_long_period': 21,
         'ema_short_period': 12,
         'momentum_multiplier': 0.5,
         'momentum_period': 10,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.5,
         'volatility_squeeze_threshold': 0.1,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
