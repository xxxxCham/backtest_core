from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_stoch_rsi_mean_reversion_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "warmup": 100,
            "atr_period": 14,
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "stoch_rsi_period": 14,
            "stoch_rsi_signal_period": 9,
            "stop_atr_mult": 2.0,
            "tp_atr_mult": 5.0,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "leverage": ParameterSpec(type=float, default=1.0, min_val=1.0, max_val=2.0, description="Position sizing"),
            "warmup": ParameterSpec(type=int, default=100, min_val=50, max_val=200, description="Data warmup"),
            "atr_period": ParameterSpec(type=int, default=14, min_val=10, max_val=25, description="ATR calculation period"),
            "bollinger_period": ParameterSpec(type=int, default=20, min_val=15, max_val=30, description="Bollinger period"),
            "bollinger_std_dev": ParameterSpec(type=float, default=2.0, min_val=1.5, max_val=3.0, description="Standard deviation multiplier"),
            "stoch_rsi_period": ParameterSpec(type=int, default=14, min_val=10, max_val=20, description="Stochastic RSI period"),
            "stoch_rsi_signal_period": ParameterSpec(type=int, default=9, min_val=5, max_val=15, description="Signal period"),
            "stop_atr_mult": ParameterSpec(type=float, default=2.0, min_val=1.5, max_val=3.0, description="Stop loss multiplier"),
            "tp_atr_mult": ParameterSpec(type=float, default=5.0, min_val=3.0, max_val=7.0, description="Take profit multiplier"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract and sanitize indicator arrays
        bb = indicators["bollinger"]
        bollinger_lower = np.nan_to_num(bb["lower"]).reshape(-1)
        bollinger_upper = np.nan_to_num(bb["upper"]).reshape(-1)
        bollinger_mean = np.nan_to_num(bb["middle"]).reshape(-1)
        
        srsi = indicators["stoch_rsi"]
        k_val = np.nan_to_num(srsi["k"]).reshape(-1)
        d_val = np.nan_to_num(srsi["d"]).reshape(-1)
        signal_val = np.nan_to_num(srsi["signal"]).reshape(-1)
        
        atr_val = np.nan_to_num(indicators["atr"]).reshape(-1)
        close_prices = df["close"].values
        
        # Warmup protection
        if len(df) < params["warmup"]:
            return signals
        
        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        
        # Boolean masks for valid data
        valid_lower = ~np.isnan(bollinger_lower)
        valid_upper = ~np.isnan(bollinger_upper)
        valid_mean = ~np.isnan(bollinger_mean)
        valid_k = ~np.isnan(k_val)
        valid_signal = ~np.isnan(signal_val)
        
        # Long entry conditions
        condition1 = (close_prices <= bollinger_lower)
        condition2 = (k_val < 20)
        long_entry = (condition1 & condition2 & valid_lower & valid_k)
        
        # Long exit conditions
        conditionA = (bollinger_lower <= close_prices) & (close_prices <= bollinger_mean)
        conditionB = (signal_val >= 50)
        long_exit = (conditionA | conditionB) & valid_mean & valid_signal
        
        # Apply long signals
        signals[long_entry] = 1.0
        long_entry_mask = long_entry.copy()
        
        # Calculate SL/TP levels on entry bars
        df.loc[long_entry_mask, "bb_stop_long"] = close_prices[long_entry_mask] - params["stop_atr_mult"] * atr_val[long_entry_mask]
        df.loc[long_entry_mask, "bb_tp_long"] = close_prices[long_entry_mask] + params["tp_atr_mult"] * atr_val[long_entry_mask]
        
        # Short entry conditions (disabled - hypothesis specifies long only)
        short_entry = np.zeros(len(df), dtype=bool)
        
        return signals