from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_pivot_atr_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'pivot_points']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_vol_threshold': 0.0015,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 6.0,
         'trailing_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0005,
                max_val=0.005,
                default=0.0015,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=6.0,
                param_type='float',
                step=0.1,
            ),
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract needed series and apply nan handling
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        pp = indicators['pivot_points']
        pivot = np.nan_to_num(pp["pivot"])
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])

        atr_vol_threshold = params.get("atr_vol_threshold", 0.0015)

        # Entry conditions
        long_entry = (close > r1) & (atr > atr_vol_threshold)
        short_entry = (close < s1) & (atr > atr_vol_threshold)

        # Populate masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns with NaN
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan
        df["sl_level"] = np.nan
        df["tp_level"] = np.nan

        # Compute ATR‑based stop‑loss and take‑profit levels on entry bars
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 6.0)

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[long_mask, "sl_level"] = close[long_mask] - params.get("trailing_atr_mult", 2.0) * atr[long_mask]
        df.loc[long_mask, "tp_level"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        df.loc[short_mask, "sl_level"] = close[short_mask] + params.get("trailing_atr_mult", 2.0) * atr[short_mask]
        df.loc[short_mask, "tp_level"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
