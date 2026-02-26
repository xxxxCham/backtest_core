from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='moveusdc_ema_bb')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_mult_stop': 0.1,
         'atr_mult_tp': 0.25,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_mult_stop': ParameterSpec(
                name='atr_mult_stop',
                min_val=0.01,
                max_val=2.0,
                default=0.1,
                param_type='float',
                step=0.1,
            ),
            'atr_mult_tp': ParameterSpec(
                name='atr_mult_tp',
                min_val=0.05,
                max_val=4.0,
                default=0.25,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        ema = np.nan_to_num(indicators['ema'])
        close = df["close"].values
        volume = df["volume"].values
        bollinger = indicators['bollinger']
        upper = np.nan_to_num(indicators['bollinger']["upper"])
        lower = np.nan_to_num(indicators['bollinger']["lower"])

        long_mask =  (close > ema) & \
                     (close < lower) & \
                     (volume <= 10**6)  # close to zero volume

        short_mask = (close < ema) | (close > upper)  

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        signals[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
