from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Multi-factor Strategy on ALGOUSDC 15m')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'fees': 0,
         'leverage': 2,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'slippage': 0,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'fees': ParameterSpec(
                name='fees',
                min_val=0,
                max_val=0.01,
                default=0,
                param_type='float',
                step=0.1,
            ),
            'slippage': ParameterSpec(
                name='slippage',
                min_val=0,
                max_val=0.02,
                default=5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=3,
                default=2,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=6.0,
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
        # Initialize long_mask and short_mask to zeros
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)


        # Write SL/TP levels into df if using ATR-based risk management
        sl_level = pd.Series(np.nan, index=df.index)
        tp_level = pd.Series(np.nan, index=df.index)
        signals.iloc[:warmup] = 0.0
        return signals
