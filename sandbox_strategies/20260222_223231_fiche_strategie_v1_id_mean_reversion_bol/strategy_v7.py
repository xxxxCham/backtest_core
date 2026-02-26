from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_rsi_mean_reversion_with_trend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 3.0, 'tp_atr_mult': 5.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.0,
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

    def _to_array(self, value: Any, n: int) -> np.ndarray:
        """Ensure that an indicator value is a 1‑D numpy array of length n."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, arr.item())
        return arr

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Prepare empty signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # --- Indicator extraction -----------------------------------------
        close = np.nan_to_num(df["close"].values)

        bb = indicators['bollinger']
        lower = np.nan_to_num(self._to_array(bb["lower"], n))
        middle = np.nan_to_num(self._to_array(bb["middle"], n))
        upper = np.nan_to_num(self._to_array(bb["upper"], n))

        rsi = np.nan_to_num(self._to_array(indicators['rsi'], n))
        adx_arr = np.nan_to_num(self._to_array(indicators['adx']["adx"], n))
        atr = np.nan_to_num(self._to_array(indicators['atr'], n))

        # --- Helper for cross detection ------------------------------------
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # --- Long / short entry masks -------------------------------------
        long_mask = (close < lower) & (rsi < 30) & (adx_arr < 20)
        short_mask = (close > upper) & (rsi > 70) & (adx_arr < 20)

        # --- Exit mask -----------------------------------------------------
        exit_mask = (
            cross_any(close, middle)
            | cross_any(rsi, np.full(n, 50.0))
            | (adx_arr > 25)
        )

        # --- Apply signals -------------------------------------------------
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # --- Warmup protection ---------------------------------------------
        signals.iloc[:warmup] = 0.0

        # --- Prepare SL/TP columns -----------------------------------------
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 3.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.0))

        # ATR‑based levels on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals