from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_21_pullback_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"ema_period": 21, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_period": ParameterSpec(
                type=int,
                bounds=(2, 100),
                default=21,
                description="Period for EMA calculation"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                bounds=(2, 50),
                default=14,
                description="Period for RSI calculation"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                bounds=(0.5, 3),
                default=1.5,
                description="Stop loss multiplier for ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                bounds=(1, 4),
                default=3.0,
                description="Take profit multiplier for ATR"
            ),
            "warmup": ParameterSpec(
                type=int,
                bounds=(20, 100),
                default=50,
                description="Number of warmup bars to avoid initial NaN signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Warmup period to avoid NaN signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Get indicators with proper access
        ema = np.nan_to_num(indicators["ema"][params["ema_period"]])
        rsi = np.nan_to_num(indicators["rsi"][params["rsi_period"]])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        
        # Entry conditions
        long_entry = (df.close > ema) & (rsi < 70)
        short_entry = (df.close < ema) & (rsi > 30)
        
        # Exit conditions
        price_cross_ema = df.close < ema
        rsi_cross_70 = rsi > 70
        
        # Signal logic
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0
        
        # Close positions when exit conditions met
        signals[price_cross_ema] = 0.0
        signals[rsi_cross_70] = 0.0
        
        # Stop loss and take profit logic
        atr = np.nan_to_num(indicators["atr"])
        stop_level = df.close - params["stop_atr_mult"] * atr
        tp_level_long = df.close + params["tp_atr_mult"] * atr
        tp_level_short = df.close - params["tp_atr_mult"] * atr
        
        # Check for stop loss and take profit conditions
        signals[(df.close < stop_level)] = 0.0
        signals[(df.close > tp_level_long)] = 0.0
        signals[(df.close > stop_level)] = 0.0
        signals[(df.close < tp_level_short)] = 0.0
        
        return signals