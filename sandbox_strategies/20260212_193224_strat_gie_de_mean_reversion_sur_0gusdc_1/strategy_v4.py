from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v4")

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
        
        cci_period = int(params.get("cci_period", 14))
        cci_threshold = float(params.get("cci_threshold", 100))
        keltner_multiplier = float(params.get("keltner_multiplier", 1.5))
        keltner_period = int(params.get("keltner_period", 20))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        warmup = int(params.get("warmup", 50))
        
        close = np.nan_to_num(df['close'].values)
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        keltner = indicators["keltner"]
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # Entry condition: price below lower Keltner channel and CCI shows reversal
        entry_long = (close < keltner_lower) & (cci < -cci_threshold) & (cci > np.roll(cci, 1))
        
        # Exit condition: price crosses above middle Keltner channel
        exit_long = (close > keltner_middle) & (np.roll(close, 1) < keltner_middle)
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0  # LONG signal
                
                # Apply stop-loss and take-profit logic
                stop_price = close[i] - stop_atr_mult * atr[i]
                take_profit_price = close[i] + tp_atr_mult * atr[i]
                
                # Find next exit
                for j in range(i + 1, len(signals)):
                    if close[j] <= stop_price or close[j] >= take_profit_price:
                        signals.iloc[j] = 0.0  # FLAT
                        break
                    if j >= len(signals) - 1:
                        signals.iloc[j] = 0.0  # FLAT at end
        
        signals.iloc[:warmup] = 0.0
        return signals