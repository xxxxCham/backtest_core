from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v5")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "cci_threshold": 100, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(10, 30, 1),
            "cci_threshold": ParameterSpec(50, 150, 10),
            "keltner_multiplier": ParameterSpec(1.0, 2.0, 0.1),
            "keltner_period": ParameterSpec(10, 30, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
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
        cci = indicators["cci"]
        atr = indicators["atr"]
        
        # Compute Keltner channel components
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Compute CCI
        cci = np.nan_to_num(cci)
        
        # Compute ATR
        atr = np.nan_to_num(atr)
        
        # Entry conditions
        close = np.nan_to_num(df["close"].values)
        cci_threshold = params["cci_threshold"]
        
        # CCI reversal: from negative to positive
        cci_prev = np.roll(cci, 1)
        cci_reversal = (cci < -cci_threshold) & (cci_prev > -cci_threshold)
        
        # Price below lower Keltner channel
        price_below_lower = close < keltner_lower
        
        # Momentum confirmation: price above previous close
        close_prev = np.roll(close, 1)
        momentum_confirm = close > close_prev
        
        # Entry long condition
        entry_long = price_below_lower & cci_reversal & momentum_confirm
        
        # Exit condition: price crosses middle Keltner channel
        middle_prev = np.roll(keltner_middle, 1)
        exit_condition = (close > keltner_middle) & (close_prev < keltner_middle)
        
        # Set signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_condition)[0]
        
        for idx in entry_indices:
            signals.iloc[idx] = 1.0  # Long signal
            
        # Set exit signals
        for idx in exit_indices:
            if signals.iloc[idx] == 1.0:
                signals.iloc[idx] = 0.0  # Flat signal
                
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals