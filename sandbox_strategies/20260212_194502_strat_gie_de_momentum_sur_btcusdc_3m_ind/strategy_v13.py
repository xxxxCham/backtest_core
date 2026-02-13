from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_momentum_three_indicator")

    @property
    def required_indicators(self) -> List[str]:
        return ["macd", "roc", "atr", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "bb_period": 20, "bb_std_dev": 2, "macd_fast": 12, "macd_signal": 9, "macd_slow": 26, "roc_period": 10, "stop_atr_mult": 2.0, "tp_atr_mult": 4.5, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "bb_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
            "bb_std_dev": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "macd_fast": ParameterSpec(param_type="int", min_value=5, max_value=20, step=1),
            "macd_signal": ParameterSpec(param_type="int", min_value=3, max_value=15, step=1),
            "macd_slow": ParameterSpec(param_type="int", min_value=15, max_value=50, step=1),
            "roc_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        macd = indicators["macd"]
        roc = np.nan_to_num(indicators["roc"])
        atr = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        
        # Get required arrays
        macd_hist = np.nan_to_num(macd["histogram"])
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        long_condition = (macd_hist > 0) & (roc > 0) & (close > bb_upper)
        short_condition = (macd_hist < 0) & (roc < 0) & (close < bb_lower)
        
        # Exit conditions
        macd_hist_shift = np.roll(macd_hist, 1)
        roc_shift = np.roll(roc, 1)
        exit_long = (macd_hist * macd_hist_shift < 0) | (roc * roc_shift < 0)
        exit_short = (macd_hist * macd_hist_shift < 0) | (roc * roc_shift < 0)
        
        # Generate signals
        long_signals = long_condition & ~exit_long
        short_signals = short_condition & ~exit_short
        
        # Convert to signals
        signals[long_signals] = 1.0
        signals[short_signals] = -1.0
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals