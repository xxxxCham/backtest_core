from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='regime_adaptive_obv_atr_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 25,
         'atr_threshold': 1.0,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 20}

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
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
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
                max_val=5.0,
                default=2.4,
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

        # Wrap indicator arrays
        atr = np.nan_to_num(indicators['atr'])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        obv_arr = np.nan_to_num(indicators['obv'])
        close_arr = df["close"].values

        # Previous values for OBV and close
        prev_obv = np.roll(obv_arr, 1)
        prev_obv[0] = np.nan
        prev_close = np.roll(close_arr, 1)
        prev_close[0] = np.nan

        # Volatility regimes
        high_vol = atr > params["atr_threshold"]
        low_vol = ~high_vol

        # ADX strength
        strong_adx = adx_arr > params["adx_threshold"]
        weak_adx = adx_arr < 20

        # OBV trend
        obv_rising = obv_arr > prev_obv
        obv_falling = obv_arr < prev_obv

        # Entry long conditions
        long_high = high_vol & strong_adx & obv_rising
        long_low = low_vol & weak_adx & obv_rising & (close_arr < prev_close)
        long_mask = long_high | long_low

        # Entry short conditions
        short_high = high_vol & strong_adx & obv_falling
        short_low = low_vol & weak_adx & obv_falling & (close_arr > prev_close)
        short_mask = short_high | short_low

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic
        obv_cross_down = obv_arr < prev_obv
        obv_cross_up = obv_arr > prev_obv
        long_exit = (atr < params["atr_threshold"]) | obv_cross_down
        short_exit = (atr < params["atr_threshold"]) | obv_cross_up

        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Additional hard warmup to avoid NaN false signals
        signals.iloc[:50] = 0.0

        # SL/TP columns for ATR‑based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = (
            close_arr[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        )
        df.loc[entry_long_mask, "bb_tp_long"] = (
            close_arr[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        )
        df.loc[entry_short_mask, "bb_stop_short"] = (
            close_arr[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        )
        df.loc[entry_short_mask, "bb_tp_short"] = (
            close_arr[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        )
        signals.iloc[:warmup] = 0.0
        return signals
