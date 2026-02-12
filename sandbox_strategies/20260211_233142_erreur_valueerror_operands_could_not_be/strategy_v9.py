from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_with_risk")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(type=float, min_val=50, max_val=90),
            "rsi_oversold": ParameterSpec(type=float, min_val=10, max_val=50),
            "rsi_period": ParameterSpec(type=int, min_val=2, max_val=50),
            "stop_atr_mult": ParameterSpec(type=float, min_val=1.0, max_val=3.0),
            "tp_atr_mult": ParameterSpec(type=float, min_val=2.0, max_val=6.0),
            "warmup": ParameterSpec(type=int, min_val=0, max_val=100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0
        
        close_price = np.nan_to_num(df['close'].values.astype(float))
        rsi_values = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr_values = np.nan_to_num(indicators["atr"])

        entry_long = (close_price > (bb_upper * 0.99)) & (rsi_values < params["rsi_oversold"])
        entry_short = (close_price < (bb_lower * 1.01)) & (rsi_values > params["rsi_overbought"])

        stop_loss_long = close_price - (atr_values * params["stop_atr_mult"])
        take_profit_long = close_price + (atr_values * params["tp_atr_mult"])
        stop_loss_short = close_price + (atr_values * params["stop_atr_mult"])
        take_profit_short = close_price - (atr_values * params["tp_atr_mult"])

        signals[entry_long] = 1.0
        signals[entry_short] = -1.0

        return signals