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
        return {'rsi_overbought': 100,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.0,
         'warmup': 0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get('warmup', 50))
        rsi_oversold = float(params.get('rsi_oversold', 30))
        rsi_overbought = float(params.get('rsi_overbought', 70))
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < warmup + 2:
            return signals
        rsi_raw = indicators.get('rsi')
        bb_raw = indicators.get('bollinger')
        has_rsi = isinstance(rsi_raw, np.ndarray)
        has_bb = isinstance(bb_raw, dict)
        if has_rsi:
            rsi = np.nan_to_num(rsi_raw.astype(np.float64))
        else:
            rsi = np.full(n, 50.0)
        if has_bb:
            bb_lower = np.nan_to_num(bb_raw.get('lower', np.zeros(n)).astype(np.float64))
            bb_upper = np.nan_to_num(bb_raw.get('upper', np.zeros(n)).astype(np.float64))
        else:
            bb_lower = np.full(n, 0.0)
            bb_upper = np.full(n, np.inf)
        long_mask = (rsi < rsi_oversold) & (close <= bb_lower)
        short_mask = (rsi > rsi_overbought) & (close >= bb_upper)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
