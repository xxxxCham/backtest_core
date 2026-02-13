from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BTCUSDC_30m_Breakout_Strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec(5, 30, 1),
            "keltner_multiplier": ParameterSpec(0.5, 3.0, 0.1),
            "keltner_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec(1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec(5, 30, 1),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        keltner = indicators["keltner"]
        supertrend = indicators["supertrend"]
        atr = indicators["atr"]
        
        # Get arrays with NaN handling
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        supertrend_value = np.nan_to_num(supertrend["supertrend"])
        supertrend_direction = np.nan_to_num(supertrend["direction"])
        atr_values = np.nan_to_num(atr)
        
        # Get close prices
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions for short trades
        # Price closes above Keltner upper band
        price_above_upper = close > keltner_upper
        # Supertrend is in uptrend (direction > 0)
        uptrend_condition = supertrend_direction > 0
        
        # Combine entry conditions
        entry_short = price_above_upper & uptrend_condition
        
        # Exit conditions
        # Price re-enters Keltner range
        reenters_range = (close > keltner_lower) & (close < keltner_upper)
        
        # Find entry points
        entry_points = np.where(entry_short)[0]
        
        # Initialize exit signals
        exit_points = set()
        
        # For each entry point, find exit
        for i in entry_points:
            # Set initial stop loss and take profit
            entry_price = close[i]
            atr_mult = atr_values[i]
            stop_loss = entry_price + (params["stop_atr_mult"] * atr_mult)
            take_profit = entry_price - (params["tp_atr_mult"] * atr_mult)
            
            # Look ahead for exit conditions
            for j in range(i + 1, len(close)):
                current_price = close[j]
                
                # Exit if price re-enters range
                if reenters_range[j]:
                    exit_points.add(j)
                    break
                
                # Exit if stop loss is hit
                if current_price >= stop_loss:
                    exit_points.add(j)
                    break
                
                # Exit if take profit is hit
                if current_price <= take_profit:
                    exit_points.add(j)
                    break
        
        # Generate signals
        for i in entry_points:
            signals.iloc[i] = -1.0  # Short signal
        
        # Mark exit points
        for i in exit_points:
            signals.iloc[i] = 0.0  # Flat signal
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals