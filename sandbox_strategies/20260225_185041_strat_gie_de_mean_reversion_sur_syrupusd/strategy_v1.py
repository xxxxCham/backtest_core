from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="cci_stoch_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["cci", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 14,
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
                max_val=30,
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
            "stoch_rsi_overbought": ParameterSpec(
                name="stoch_rsi_overbought",
                min_val=70,
                max_val=90,
                default=80,
                param_type="int",
                step=1,
            ),
            "stoch_rsi_oversold": ParameterSpec(
                name="stoch_rsi_oversold",
                min_val=10,
                max_val=30,
                default=20,
                param_type="int",
                step=1,
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=5,
                max_val=20,
                default=14,
                param_type="int",
                step=1,
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
        n = len(df)
        warmup = int(params.get("warmup", 50))

        # Initialise signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays safely
        cci = np.nan_to_num(indicators['cci'])
        stoch_k = np.nan_to_num(indicators['stoch_rsi']["k"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_cond = (cci < -100) & (stoch_k < params["stoch_rsi_oversold"])
        short_cond = (cci > 100) & (stoch_k > params["stoch_rsi_overbought"])
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0

        # Exit when CCI crosses zero
        prev_cci = np.roll(cci, 1)
        prev_cci[0] = np.nan  # first element has no previous value
        exit_mask = ((cci > 0) & (prev_cci <= 0)) | ((cci < 0) & (prev_cci >= 0))
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based stop/target columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - (
            params["stop_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + (
            params["tp_atr_mult"] * atr[entry_long]
        )
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + (
            params["stop_atr_mult"] * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - (
            params["tp_atr_mult"] * atr[entry_short]
        )

        return signals