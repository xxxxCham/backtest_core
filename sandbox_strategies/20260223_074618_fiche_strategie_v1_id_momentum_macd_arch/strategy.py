from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_macd_atr_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

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
                min_val=1.0,
                max_val=5.0,
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
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x < y) & (px >= py)

        # extract indicator arrays
        close = df["close"].values
        boll = indicators['bollinger']
        upper = np.nan_to_num(boll["upper"])
        middle = np.nan_to_num(boll["middle"])
        lower = np.nan_to_num(boll["lower"])

        macd_dict = indicators['macd']
        macd_line = np.nan_to_num(macd_dict["macd"])
        signal_line = np.nan_to_num(macd_dict["signal"])
        hist = np.nan_to_num(macd_dict["histogram"])

        atr_arr = np.nan_to_num(indicators['atr'])

        # entry conditions
        long_cross = cross_up(close, upper)
        short_cross = cross_down(close, lower)
        macd_long_cross = cross_up(macd_line, signal_line)
        macd_short_cross = cross_down(macd_line, signal_line)

        # ATR filter – use a simple positive check (adjust if needed)
        atr_filter = atr_arr > 0

        long_mask = long_cross & macd_long_cross & atr_filter
        short_mask = short_cross & macd_short_cross & atr_filter

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        exit_cross = cross_down(close, middle)

        # histogram sign change detection
        hist_sign_change = np.logical_xor(hist > 0, np.roll(hist, 1) > 0)
        hist_sign_change[0] = False

        exit_mask = exit_cross | hist_sign_change
        signals[exit_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # risk management columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        # long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr_arr[long_mask]

        # short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr_arr[short_mask]

        return signals