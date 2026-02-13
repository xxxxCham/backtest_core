from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        middle_bb = np.nan_to_num(bb["middle"])
        upper_bb = np.nan_to_num(bb["upper"])
        atr = np.nan_to_num(indicators["atr"])
        ema_50 = np.nan_to_num(indicators["ema"])
        
        # Entry conditions
        # Price touches lower Bollinger Band
        price_touches_lower = np.isclose(df["close"].values, lower_bb, rtol=1e-5)
        # RSI below oversold
        rsi_below_oversold = rsi < params["rsi_oversold"]
        # RSI is rising
        rsi_shifted = np.roll(rsi, 1)
        rsi_rising = rsi > rsi_shifted
        # Price above 50-period EMA
        price_above_ema = df["close"].values > ema_50
        
        # Combine all entry conditions
        entry_condition = (
            price_touches_lower &
            rsi_below_oversold &
            rsi_rising &
            price_above_ema
        )
        
        # Exit when price crosses above middle Bollinger Band
        price_crosses_middle = df["close"].values > middle_bb
        exit_condition = price_crosses_middle
        
        # Generate signals
        entry_indices = np.where(entry_condition)[0]
        exit_indices = np.where(exit_condition)[0]
        
        # Set signals
        for i in entry_indices:
            signals.iloc[i] = 1.0  # LONG signal
            
        # Simple exit logic: when price crosses above middle BB, flat
        for i in exit_indices:
            if signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0  # FLAT
                
        return signals