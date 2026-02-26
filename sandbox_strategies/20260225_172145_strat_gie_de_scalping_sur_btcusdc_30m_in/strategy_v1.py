from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='btc_usdc_scalp_ema_bollinger_vwap')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is required for risk management
        return ['ema', 'bollinger', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'bollinger_period': 20,
            'bollinger_std_dev': 2,
            'ema_period': 20,
            'leverage': 1,
            'stop_atr_mult': 1.5,
            'tp_atr_mult': 3.9,
            'warmup': 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1,
                max_val=3,
                default=2,
                param_type='float',
                step=0.1,
            ),
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
                max_val=6.0,
                default=3.9,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Helper cross functions
        def cross_up(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Extract indicator arrays
        close = df["close"].values
        ema_arr = np.nan_to_num(indicators['ema'])
        vwap_arr = np.nan_to_num(indicators['vwap'])
        atr_arr = np.nan_to_num(indicators['atr'])
        boll_upper = np.nan_to_num(indicators['bollinger']["upper"])
        boll_lower = np.nan_to_num(indicators['bollinger']["lower"])

        # Long entry conditions
        long_cond = (
            (close > boll_upper)
            & (ema_arr > vwap_arr)
            & cross_up(close, ema_arr)
        )
        # Short entry conditions
        short_cond = (
            (close < boll_lower)
            & (ema_arr < vwap_arr)
            & cross_down(close, ema_arr)
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        if np.any(long_cond):
            df.loc[long_cond, "bb_stop_long"] = close[long_cond] - params["stop_atr_mult"] * atr_arr[long_cond]
            df.loc[long_cond, "bb_tp_long"] = close[long_cond] + params["tp_atr_mult"] * atr_arr[long_cond]

        if np.any(short_cond):
            df.loc[short_cond, "bb_stop_short"] = close[short_cond] + params["stop_atr_mult"] * atr_arr[short_cond]
            df.loc[short_cond, "bb_tp_short"] = close[short_cond] - params["tp_atr_mult"] * atr_arr[short_cond]

        return signals