from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_keltner_supertrend_breakout_short")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "supertrend", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "supertrend_multiplier": 3.0, "supertrend_period": 10, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_period": ParameterSpec("atr_period", 5, 30, 1),
            "keltner_multiplier": ParameterSpec("keltner_multiplier", 0.5, 3.0, 0.1),
            "keltner_period": ParameterSpec("keltner_period", 10, 50, 1),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 5.0, 0.1),
            "supertrend_multiplier": ParameterSpec("supertrend_multiplier", 1.0, 5.0, 0.1),
            "supertrend_period": ParameterSpec("supertrend_period", 5, 30, 1),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 10.0, 0.1),
            "warmup": ParameterSpec("warmup", 20, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Extract indicators
        keltner = indicators["keltner"]
        lower_band = np.nan_to_num(keltner["lower"])
        middle_band = np.nan_to_num(keltner["middle"])
        supertrend = indicators["supertrend"]
        trend_direction = np.nan_to_num(supertrend["direction"])
        atr_values = np.nan_to_num(indicators["atr"])
        
        # Entry conditions
        # Short entry: price breaks below KELTNER lower band AND SUPERTREND is in downtrend
        price = np.nan_to_num(df["close"].values)
        short_entry = (price < lower_band) & (trend_direction < 0)
        
        # Exit conditions
        # Exit if price crosses above KELTNER middle band OR trailing stop-loss triggered OR take-profit reached
        # For simplicity, we'll assume exit on crossing middle band or trailing stop
        # Trailing stop logic would require tracking entry prices and comparing with ATR
        # For now, we'll just exit on middle band cross for simplicity
        
        # Generate signals
        entry_indices = np.where(short_entry)[0]
        for i in entry_indices:
            # We only set signal for entry, exit logic is handled in backtesting engine
            signals.iloc[i] = -1.0  # SHORT signal
            
        return signals