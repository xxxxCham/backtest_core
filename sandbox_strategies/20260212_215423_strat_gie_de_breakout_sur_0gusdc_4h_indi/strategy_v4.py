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
        return ['rsi', 'bollinger', 'atr']

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
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get('warmup', 50))
        rsi_oversold = float(params.get('rsi_oversold', 25))
        rsi_overbought = float(params.get('rsi_overbought', 75))
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < warmup + 2:
            return signals
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw.astype(np.float64))
        else:
            rsi = np.full(n, 50.0)
        # Long-only: enter when RSI crosses above oversold, exit at overbought
        long_mask = np.zeros(n, dtype=bool)
        for j in range(warmup + 1, n):
            if rsi[j] > rsi_oversold and rsi[j - 1] <= rsi_oversold:
                long_mask[j] = True
        exit_mask = rsi > rsi_overbought
        signals[long_mask] = 1.0
        signals[exit_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
