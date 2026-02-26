from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.75,
         'tp_atr_mult': 6.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=6.0,
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
        # REQUIRED INDICATORS:
        required_indicators = ['bollinger', 'ema', 'rsi']

        # DEFAULT PARAMETERS:
        default_params = {
            "bb_atr_mult": 1.0,
            "bb_tp_atr_mult": 2.0,
        }

        # FUNCTION BODY:
        def generate_signals(df):

            # Initialize signals and other variables
            long_mask = np.zeros(len(df), dtype=bool)
            short_mask = np.zeros(len(df), dtype=bool)

            # Read indicators
            bollinger = indicators['bollinger']
            upper, middle, lower = indicators['bollinger']["upper"], indicators['bollinger']["middle"], indicators['bollinger']["lower"]
            ema = np.nan_to_num(indicators['ema'])
            rsi = np.nan_to_num(indicators['rsi'])

            # Create signals
            long_mask |= (close > upper) & (close < middle)
            short_mask |= (close < lower) | (rsi < 30.0)

            entry_price = close.copy()
            df.loc[:, "bb_stop_long"] = entry_price - default_params["bb_atr_mult"] * atr
            df.loc[:, "bb_tp_long"]   = entry_price + default_params["bb_tp_atr_mult"] * atr

            # Filter out NaNs and generate signals
            long_mask &= ~np.isnan(entry_price)
            short_mask &= ~np.isnan(entry_price)

            # Set signals
            signals.iloc[:50] = 0.0  # Skip warmup phase: all signals are NaN until now
            signals[long_mask] = 1.0
            signals[short_mask] = -1.0

            return long_mask, short_mask
        signals.iloc[:warmup] = 0.0
        return signals
