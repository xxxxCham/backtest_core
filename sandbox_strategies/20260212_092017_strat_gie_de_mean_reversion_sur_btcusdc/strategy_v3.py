from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_meanreversion_stoch_rsi_bollinger_short_only")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1, "Overbought level for RSI"),
            "rsi_oversold": ParameterSpec(10, 40, 1, "Oversold level for RSI"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop-loss multiplier based on ATR"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take-profit multiplier based on ATR"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup period to avoid initial NaN signals"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        bb = indicators["bollinger"]
        stoch_rsi = indicators["stoch_rsi"]
        atr = np.nan_to_num(indicators["atr"])
        
        # Ensure all arrays have the same length
        close = np.nan_to_num(df["close"].values)
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        
        # Entry conditions for short trades
        # Price below lower bollinger band
        entry_condition = close < bb_lower
        # Stochastic RSI confirms oversold conditions (k > d)
        confirmation_condition = stoch_rsi_k > stoch_rsi_d
        
        # Combine conditions
        short_entry = entry_condition & confirmation_condition
        
        # Exit condition - price crosses back above lower band
        exit_condition = close > bb_lower
        
        # Create signal series
        short_signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        short_signals.loc[short_entry] = -1.0  # Short signal
        short_signals.loc[exit_condition] = 0.0  # Flat signal on exit
        
        # Apply warmup protection
        warmup = int(params.get("warmup", 50))
        short_signals.iloc[:warmup] = 0.0
        
        signals = short_signals
        
        return signals