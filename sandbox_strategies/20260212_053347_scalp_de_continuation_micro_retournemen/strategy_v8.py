from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="structural revamp")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'rsi', 'bollinger', 'atr', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_periods': [9, 21],
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'rsi_period': 14,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 3.0,
         'trailing_stop': 0.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        def _align_len(arr: np.ndarray, target_len: int, fill: float = np.nan) -> np.ndarray:
            out = np.full(target_len, fill, dtype=np.float64)
            m = min(target_len, len(arr))
            if m > 0:
                out[:m] = arr[:m]
            return out
        close = _align_len(np.nan_to_num(df['close'].values.astype(np.float64)), n, fill=0.0)
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = _align_len(np.nan_to_num(rsi_raw.astype(np.float64)), n)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
