from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase_Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'atr', 'stochastic', 'mfi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'RSI_period': 14,
         'STOCHASTIC_fast_k_ period': 5,
         'STOCHASTIC_slow_k_period': 30,
         'leverage': 1,
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
        # Initialize mask variables
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # ATR-based stop loss and take profit logic
        atr = indicators['atr']
        ema = indicators['ema']
        macd = indicators['macd']
        stochastic = indicators['stochastic']
        mfi = indicators['mfi']


        # Write SL/TP columns into df if using ATR-based risk management
        sl_level, tp_level = params.get("sl_level"), params.get("tp_level")
        if sl_level and tp_level:
            atr = indicators['atr']
            ema = indicators['ema']

            signals[signals == 1] = np.nan
        signals.iloc[:warmup] = 0.0
        return signals
