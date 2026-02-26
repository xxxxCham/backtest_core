from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.25, 'tp_atr_mult': 4.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=4.5,
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
        # Masks for long/short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays safely
        supertrend_dict = indicators['supertrend']
        adx_dict = indicators['adx']
        atr_arr = np.nan_to_num(indicators['atr'])
        close_arr = df["close"].values

        direction = np.nan_to_num(supertrend_dict["direction"])
        supertrend_val = np.nan_to_num(supertrend_dict["supertrend"])
        adx_val = np.nan_to_num(adx_dict["adx"])

        # Long entry: direction 1, strong ADX, price above supertrend band
        long_mask = (
            (direction == 1)
            & (adx_val > 30)
            & (close_arr > supertrend_val)
        )

        # Short entry: direction -1, strong ADX, price below supertrend band
        short_mask = (
            (direction == -1)
            & (adx_val > 30)
            & (close_arr < supertrend_val)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 2.25))
        tp_mult = float(params.get("tp_atr_mult", 4.5))

        # Long ATR SL/TP on entry bars
        df.loc[long_mask, "bb_stop_long"] = (
            close_arr[long_mask] - stop_mult * atr_arr[long_mask]
        )
        df.loc[long_mask, "bb_tp_long"] = (
            close_arr[long_mask] + tp_mult * atr_arr[long_mask]
        )

        # Short ATR SL/TP on entry bars
        df.loc[short_mask, "bb_stop_short"] = (
            close_arr[short_mask] + stop_mult * atr_arr[short_mask]
        )
        df.loc[short_mask, "bb_tp_short"] = (
            close_arr[short_mask] - tp_mult * atr_arr[short_mask]
        )
        signals.iloc[:warmup] = 0.0
        return signals
