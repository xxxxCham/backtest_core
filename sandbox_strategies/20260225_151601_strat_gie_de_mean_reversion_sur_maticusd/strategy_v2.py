from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='matic_usdc_bollinger_stochrsi_mean_rev')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'stoch_rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'leverage': 1,
            'overbought_threshold': 80,
            'oversold_threshold': 20,
            'stoch_rsi_period': 14,
            'stop_atr_mult': 1.2,
            'tp_atr_mult': 2.9,
            'warmup': 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std': ParameterSpec(
                name='bollinger_std',
                min_val=1,
                max_val=3,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'stoch_rsi_period': ParameterSpec(
                name='stoch_rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'overbought_threshold': ParameterSpec(
                name='overbought_threshold',
                min_val=70,
                max_val=90,
                default=80,
                param_type='int',
                step=1,
            ),
            'oversold_threshold': ParameterSpec(
                name='oversold_threshold',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=4.0,
                default=2.9,
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
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])

        srsi = indicators['stoch_rsi']
        k = np.nan_to_num(srsi["k"])

        atr = np.nan_to_num(indicators['atr'])

        # Helper functions for cross detection that accept scalar or array thresholds
        def cross_up(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            y_arr = y if isinstance(y, np.ndarray) else np.full_like(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y_arr) & (prev_x <= prev_y)

        def cross_down(x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
            y_arr = y if isinstance(y, np.ndarray) else np.full_like(x, y)
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y_arr, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y_arr) & (prev_x >= prev_y)

        # Entry conditions
        long_mask = (close <= lower) & (k <= params["oversold_threshold"])
        short_mask = (close >= upper) & (k >= params["overbought_threshold"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        long_exit_mask = (close > middle) | cross_up(k, 50.0)
        short_exit_mask = (close < middle) | cross_down(k, 50.0)

        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # ATR based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]

        signals.iloc[:warmup] = 0.0
        return signals