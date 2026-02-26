from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'entry_slippage': 2,
         'exit_slippage': 3,
         'leverage': 1,
         'roc_length': 14,
         'roc_threshold': 0.8,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_length': ParameterSpec(
                name='roc_length',
                min_val=7,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'roc_threshold': ParameterSpec(
                name='roc_threshold',
                min_val=0.8,
                max_val=1.2,
                default=0.8,
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
        def _calculate_sl_levels(df, mask):
            atr = indicators['atr']  # computed in StrategyBase's __init__

            sl_multiplier = 2./3. if params['leverage'] < 1 else 2/params['leverage']

            close = df["close"].values
            long_sl = close - (close * sl_multiplier) + atr[mask] # set SL at a certain % below closing price
            short_sl = -(close / sl_multiplier) + atr[mask]  # set TP at a certain % above closing price

            return np.minimum(long_sl, short_sl).values
        signals.iloc[:warmup] = 0.0
        return signals
