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
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=0.5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)  # Initialize boolean mask for long positions
        short_mask = np.zeros(n, dtype=bool)  # Initialize boolean mask for short positions

        if "warmup" in params:
            signals[:warmup] = 0.0  # Set initial signal values to zeros until the warm-up period is over

        rsi_val = []  # Initialize list for RSI values

        # Go through each bar in the dataframe
        for i, row in df.iterrows():
            close = row["close"]

            if "rsi" not in indicators:  # If no RSI input, skip this iteration
                continue

            rsi_val.append(indicators["rsi"].compute_value(row))  # Compute the RSI value for the current bar

        bb_atr = np.zeros(n)  # Initialize boolean mask for Bollinger Bands values (initialized to zeros)
        boll_u = np.zeros(n)
        boll_d = np.zeros(n)

        if "bollinger" not in indicators:  # If no Bollinger Bands input, skip this iteration
            continue

        atr_val = indicators["atr"].compute_value(row)  # Compute the ATR value for the current bar (already done in default params)
        boll_u[:] = close + 2 * atr_val  # Calculate upper Bollinger Bands values
        boll_d[:] = close - 2 * atr_val  # Calculate lower Bollinger Bands values

        for i, row in df.iterrows():
            close_price = row["close"]

            if "rsi" not in indicators:  # If no RSI input, skip this iteration
                continue

            rsi_val.append(indicators["rsi"].compute_value(row))  # Compute the RSI value for the current bar

        bb_atr = np.zeros(n)  # Initialize boolean mask for Bollinger Bands values (initialized to zeros)
        return signals
