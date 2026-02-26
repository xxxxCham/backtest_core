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
        return ['rsi', 'ema', 'atr', 'adx']

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
        
        # Extract and sanitize indicator arrays
        ema_10 = np.nan_to_num(indicators['ema'])
        ema_30 = np.nan_to_num(indicators['ema'])
        adx_value = np.nan_to_num(indicators['adx']['adx'])
        atr_value = np.nan_to_num(indicators['atr'])

        # Warmup protection
        start_idx = warmup

        # Create lagged arrays for proper shift operations
        prev_ema_10 = np.roll(ema_10, 1)
        prev_ema_10[0] = np.nan
        prev_ema_30 = np.roll(ema_30, 1)
        prev_ema_30[0] = np.nan

        # Momentum condition: EMA(10) > EMA(30)
        momentum_condition = (ema_10 > ema_30)[start_idx:]
        
        # ADX filter: >= 25
        adx_filter = (adx_value >= 25)[start_idx:]

        # Long entry condition
        long_entry = momentum_condition & adx_filter
        long_mask = np.zeros(n, dtype=bool)
        long_mask[start_idx:] = long_entry

        # Apply long signals
        signals[long_mask] = 1.0

        # Initialize SL/TP columns
        df = df.copy()
        df.loc[:, "sl_level"] = np.nan
        df.loc[:, "tp_level"] = np.nan

        # Calculate ATR-based levels on entry bars
        entry_mask = (signals == 1.0)
        close_prices = df["close"].values

        # Long positions handling
        long_positions = signals == 1.0
        if np.any(long_positions):
            entry_points = close_prices[long_positions]
            df.loc[long_positions, "sl_level"] = entry_points - params["stop_atr_mult"] * atr_value[long_positions]
            df.loc[long_positions, "tp_level"] = entry_points + params["tp_atr_mult"] * atr_value[long_positions]

        signals.iloc[:warmup] = 0.0
        return signals