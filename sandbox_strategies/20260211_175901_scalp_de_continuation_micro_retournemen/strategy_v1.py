from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 2.4,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
            "rsi_oversold": ParameterSpec(min_val=10, max_val=45, default=35, param_type="int"),
            "rsi_overbought": ParameterSpec(min_val=55, max_val=90, default=65, param_type="int"),
            "stop_atr_mult": ParameterSpec(min_val=0.5, max_val=5.0, default=1.5, param_type="float"),
            "tp_atr_mult": ParameterSpec(min_val=0.8, max_val=8.0, default=2.4, param_type="float"),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        mean_revert_exit = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[mean_revert_exit] = 0.0

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals