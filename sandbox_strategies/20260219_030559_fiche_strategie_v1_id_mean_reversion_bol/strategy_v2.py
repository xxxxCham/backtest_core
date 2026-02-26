from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 30,
         'rsi_oversold': 70,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR': ParameterSpec(
                name='ATR',
                min_val=0,
                max_val=20,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=0,
                max_val=99,
                default=70,
                param_type='int',
                step=1,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Initialize ATR and Bollinger Bands
            atr = np.nan_to_num(indicators["atr"])
            bollinger_window = params.get("bollinger_window", 20)
            upper_band, middle_band, lower_band = indicators["bollinger"].values() / (bollinger_window * 2) - atr

            # Get the close price of the last 'warmup' bars for protection against sudden market movements at the start
            warmup = int(params.get("warmup", 50))
            signals[:warmup] = 0.0

            long_mask = np.zeros(len(df), dtype=bool)
            short_mask = np.zeros(len(df), dtype=bool)

            # Your logic to generate buy and sell signals here
        return signals
