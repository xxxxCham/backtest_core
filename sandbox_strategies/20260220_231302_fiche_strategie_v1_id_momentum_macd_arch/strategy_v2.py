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
        # implement explicit LONG / SHORT logic
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        # implement ATR-based risk management
        atr = indicators['atr']
        ema_fast = indicators['ema']
        ema_slow = indicators['ema'] * 2 - indicators['ema'] * 3

        long_mask[signals == 1] = True   # Long entry condition, replace with your logic here

        if long_mask.sum() > 0:    # If there is a signal to go long
            signals[(long_mask) & (df["close"] > ema_slow)] |= np.where(signals[(long_mask) & (df["close"].rolling(window=atr).mean().shift(-1)>ema_fast),:] == 1, True , False )   # condition for price to go long

        if short_mask.sum() > 0:    # If there is a signal to go short
            signals[(short_mask) & (df["close"] < ema_slow)] |= np.where(signals[(short_mask) & (df["close"].rolling(window=atr).mean().shift(-1)<ema_fast),:] == 1, True , False )   # condition for price to go short
        signals.iloc[:warmup] = 0.0
        return signals
