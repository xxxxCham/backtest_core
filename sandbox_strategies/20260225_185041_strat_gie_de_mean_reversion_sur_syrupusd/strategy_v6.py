from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_cci_stochrsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["cci", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_min": 0.5,
            "atr_period": 14,
            "cci_overbought": 100,
            "cci_oversold": -100,
            "cci_period": 20,
            "leverage": 1,
            "stoch_rsi_overbought": 80,
            "stoch_rsi_oversold": 20,
            "stoch_rsi_period": 14,
            "stop_atr_mult": 1.2,
            "tp_atr_mult": 2.5,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(
                name="cci_period",
                min_val=10,
                max_val=50,
                default=20,
                param_type="int",
                step=1,
            ),
            "stoch_rsi_period": ParameterSpec(
                name="stoch_rsi_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=30,
                default=14,
                param_type="int",
                step=1,
            ),
            "atr_min": ParameterSpec(
                name="atr_min",
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type="float",
                step=0.1,
            ),
            "stop_atr_mult": ParameterSpec(
                name="stop_atr_mult",
                min_val=0.5,
                max_val=3.0,
                default=1.2,
                param_type="float",
                step=0.1,
            ),
            "tp_atr_mult": ParameterSpec(
                name="tp_atr_mult",
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Wrap indicator arrays
        cci = np.nan_to_num(indicators['cci'])
        stoch_k = np.nan_to_num(indicators['stoch_rsi']["k"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_cond = (
            (cci < params["cci_oversold"])
            & (stoch_k < params["stoch_rsi_oversold"])
            & (atr > params["atr_min"])
        )
        short_cond = (
            (cci > params["cci_overbought"])
            & (stoch_k > params["stoch_rsi_overbought"])
            & (atr > params["atr_min"])
        )

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Exit conditions using cross detection
        prev_cci = np.roll(cci, 1)
        prev_cci[0] = np.nan
        prev_stoch_k = np.roll(stoch_k, 1)
        prev_stoch_k[0] = np.nan

        cross_cci_zero_up = (cci > 0) & (prev_cci <= 0)
        cross_cci_zero_down = (cci < 0) & (prev_cci >= 0)
        cross_stoch_50_up = (stoch_k > 50) & (prev_stoch_k <= 50)
        cross_stoch_50_down = (stoch_k < 50) & (prev_stoch_k >= 50)

        exit_mask = (
            cross_cci_zero_up | cross_cci_zero_down | cross_stoch_50_up | cross_stoch_50_down
        )
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP levels
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