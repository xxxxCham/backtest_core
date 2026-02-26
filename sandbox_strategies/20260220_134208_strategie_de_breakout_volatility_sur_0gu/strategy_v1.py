from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='phase_lock_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=10,
                max_val=25,
                default=14,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=15,
                max_val=30,
                default=20,
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
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        if warmup > 0:
            signals.iloc[:warmup] = 0.0
            long_mask[:warmup] = False
            short_mask[:warmup] = False

        # Extract and sanitize indicators
        atr_array = np.nan_to_num(indicators['atr'])
        ema_array = np.nan_to_num(indicators['ema'])

        # EMA(20) price level
        ema_value = ema_array
        # ATR mean (already computed)
        atr_mean = atr_array

        # Expansion condition: ATR > 1.5x mean
        expansion = (atr_array > 1.5 * atr_mean)

        # Price above EMA(20)
        price_above_ema = (df["close"] > ema_value)

        # Long entry condition
        long_entry = expansion & price_above_ema

        long_mask[long_entry] = True
        signals[long_entry] = 1.0

        # Write SL/TP levels
        close_values = df["close"].values
        entry_mask = (signals == 1.0).astype(bool)

        if np.any(entry_mask):
            df.loc[entry_mask, "sl_level"] = close_values[entry_mask] - params["stop_atr_mult"] * atr_array[entry_mask]
            df.loc[entry_mask, "tp_level"] = close_values[entry_mask] + params["tp_atr_mult"] * atr_array[entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
