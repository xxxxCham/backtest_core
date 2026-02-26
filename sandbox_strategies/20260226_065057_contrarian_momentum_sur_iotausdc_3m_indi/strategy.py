from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aroon_ema_atr_contrarian')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 20,
         'atr_period': 14,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=3.0,
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

        aroon = indicators['aroon']
        ema = indicators['ema']
        atr = indicators['atr']
        close = df["close"].values

        # Long condition
        long_condition = (indicators['aroon']["aroon_down"] < 50) & (close < ema) & (atr > 1.5 * np.nan_to_num(np.mean(atr, axis=0)))
        long_mask = long_condition

        # Short condition
        short_condition = (indicators['aroon']["aroon_up"] > 50) & (close > ema) & (atr > 1.5 * np.nan_to_num(np.mean(atr, axis=0)))
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long_aroon = (indicators['aroon']["aroon_up"] > 50)
        exit_short_aroon = (indicators['aroon']["aroon_down"] < 50)
        exit_long_ema = (close > ema)
        exit_short_ema = (close < ema)

        # Apply exits
        signals[exit_long_aroon & (signals == 1.0)] = 0.0
        signals[exit_short_aroon & (signals == -1.0)] = 0.0
        signals[exit_long_ema & (signals == 1.0)] = 0.0
        signals[exit_short_ema & (signals == -1.0)] = 0.0

        # ATR-based SL/TP (example)
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals