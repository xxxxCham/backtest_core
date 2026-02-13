from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_adx_vol combinations")

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'adx', 'obv', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 1.5,
         'ema_fast': 10,
         'ema_slow': 30,
         'leverage': 1,
         'obv_volume_period': 14,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.5,
         'warmup': 50}

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
