from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_supertrend_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
            "ema_period": 50,
            "leverage": 1,
            "stop_atr_mult": 2.75,
            "supertrend_multiplier": 3,
            "supertrend_period": 10,
            "tp_atr_mult": 5.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_period": ParameterSpec(
                name="ema_period",
                min_val=10,
                max_val=200,
                default=50,
                param_type="int",
                step=1,
            ),
            "supertrend_period": ParameterSpec(
                name="supertrend_period",
                min_val=5,
                max_val=20,
                default=10,
                param_type="int",
                step=1,
            ),
            "supertrend_multiplier": ParameterSpec(
                name="supertrend_multiplier",
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type="float",
                step=0.1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=4.0,
                default=2.75,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=10.0,
                default=5.5,
                param_type="float",
                step=0.1,
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1,
                max_val=2,
                default=1,
                param_type="int",
                step=1,
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # initialise signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # pull indicator arrays
        close = np.nan_to_num(df["close"].values, nan=0.0)
        ema = np.nan_to_num(indicators['ema'], nan=0.0)
        # supertrend direction: 1 for bullish, -1 for bearish
        st_dir = np.array(indicators['supertrend']["direction"], dtype=float)
        atr = np.nan_to_num(indicators['atr'], nan=0.0)

        # masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Entry conditions
        long_cond = (close > ema) & (st_dir == 1)
        short_cond = (close < ema) & (st_dir == -1)
        long_mask[long_cond] = True
        short_mask[short_cond] = True

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_dir = np.roll(st_dir, 1).astype(float)
        prev_dir[0] = np.nan
        dir_change_to_bearish = (st_dir == -1) & (prev_dir == 1)
        exit_long = (close < ema) | dir_change_to_bearish
        signals[exit_long] = 0.0

        dir_change_to_bullish = (st_dir == 1) & (prev_dir == -1)
        exit_short = (close > ema) | dir_change_to_bullish
        signals[exit_short] = 0.0

        # ATR-based stop/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 5.5))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # enforce warmup period
        signals.iloc[:warmup] = 0.0
        return signals