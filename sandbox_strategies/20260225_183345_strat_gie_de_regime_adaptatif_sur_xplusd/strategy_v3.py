from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='regime_adaptive_xplusdc_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_threshold': 1.0,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 3.36,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.1,
                max_val=5.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.36,
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
        # Boolean masks initialization
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        atr = np.nan_to_num(indicators['atr'])
        obv = np.nan_to_num(indicators['obv'])
        # OBV previous value
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # Price arrays
        close = df["close"].values
        open_ = df["open"].values

        # Parameters
        atr_threshold = float(params.get("atr_threshold", 1.0))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.4))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.36))

        # Entry conditions
        long_entry_mask = (atr > atr_threshold) & (close > open_) & (obv > prev_obv)
        short_entry_mask = (atr > atr_threshold) & (close < open_) & (obv < prev_obv)

        # Apply entry signals
        signals[long_entry_mask] = 1.0
        signals[short_entry_mask] = -1.0

        # Exit conditions
        exit_long_mask = (close < open_) | (atr <= atr_threshold)
        exit_short_mask = (close > open_) | (atr <= atr_threshold)

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long exit levels
        df.loc[long_entry_mask, "bb_stop_long"] = close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        df.loc[long_entry_mask, "bb_tp_long"] = close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]

        # Short exit levels
        df.loc[short_entry_mask, "bb_stop_short"] = close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        df.loc[short_entry_mask, "bb_tp_short"] = close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
