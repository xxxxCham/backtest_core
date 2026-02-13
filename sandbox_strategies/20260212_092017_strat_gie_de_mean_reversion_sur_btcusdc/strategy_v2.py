from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_meanreversion_stoch_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "stoch_rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 1),
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
        
        # Extract Bollinger bands
        bb_upper = np.nan_to_num(bb["upper"])
        bb_middle = np.nan_to_num(bb["middle"])
        bb_lower = np.nan_to_num(bb["lower"])
        
        # Extract Stochastic RSI
        stoch_rsi_k = np.nan_to_num(stoch_rsi["k"])
        stoch_rsi_d = np.nan_to_num(stoch_rsi["d"])
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        # Short only when price is below lower band and stochastic RSI confirms overbought
        entry_condition = (close <= bb_lower) & (stoch_rsi_k > stoch_rsi_d) & (stoch_rsi_k > 80)
        
        # Exit condition when price returns to middle band
        exit_condition = close >= bb_middle
        
        # Generate signals
        entry_signals = pd.Series(0.0, index=df.index)
        entry_signals[entry_condition] = -1.0  # Short signal
        
        exit_signals = pd.Series(0.0, index=df.index)
        exit_signals[exit_condition] = 0.0  # Flat signal
        
        # Combine entry and exit
        signals = entry_signals.where(entry_signals != 0, exit_signals)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals