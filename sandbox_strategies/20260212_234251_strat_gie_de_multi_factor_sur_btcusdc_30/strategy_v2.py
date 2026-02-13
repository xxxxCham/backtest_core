from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="snake_case_name")

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'stochastic', 'atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'ema_fast': 10,
         'ema_slow': 30,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stochastic_fast_period': 12,
         'stochastic_signal_period': 9,
         'stochastic_slow_period': 26,
         'stop_atr_mult': 1.0,
         'supertrend_ma_period': 26,
         'supertrend_period': 12,
         'supertrend_std_dev': 2.0,
         'tp_atr_mult': 2.0,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get('warmup', 50))
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < warmup + 2:
            return signals
        ema_raw = indicators.get('ema')
        if isinstance(ema_raw, np.ndarray):
            ema = np.nan_to_num(ema_raw.astype(np.float64))
        else:
            ema = np.full(n, 0.0)
        fast_period = int(params.get('ema_fast', 10))
        slow_period = int(params.get('ema_slow', 30))
        # Simple EMA cross using rolling means as proxy
        if n > slow_period:
            fast_ma = np.convolve(close, np.ones(fast_period)/fast_period, mode='full')[:n]
            slow_ma = np.convolve(close, np.ones(slow_period)/slow_period, mode='full')[:n]
            long_mask = (fast_ma > slow_ma) & (close > ema)
            short_mask = (fast_ma < slow_ma) & (close < ema)
            signals[long_mask] = 1.0
            signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
