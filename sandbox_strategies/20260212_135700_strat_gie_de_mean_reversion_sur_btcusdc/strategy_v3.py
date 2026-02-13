from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "keltner_multiplier": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "keltner_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
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
        keltner = indicators["keltner"]
        cci = np.nan_to_num(indicators["cci"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Keltner bands
        keltner_upper = np.nan_to_num(keltner["upper"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        keltner_middle = np.nan_to_num(keltner["middle"])
        
        # CCI series for momentum reversal detection
        cci_series = cci
        cci_prev = np.roll(cci_series, 1)
        cci_prev[0] = 0.0
        
        # Price series
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions for long
        # Price crosses below KELTNER lower band
        price_below_lower = close < keltner_lower
        # CCI transitions from positive to negative (momentum reversal)
        cci_crossing_down = (cci_prev > 0) & (cci_series <= 0)
        # Price is below KELTNER middle band
        price_below_middle = close < keltner_middle
        
        # Long entry
        long_entry = price_below_lower & cci_crossing_down & price_below_middle
        
        # Exit conditions for long
        # Price crosses back above KELTNER middle band
        price_crossing_up_middle = (np.roll(close, 1) < keltner_middle) & (close > keltner_middle)
        
        # Find entry and exit points
        entry_indices = np.where(long_entry)[0]
        exit_indices = np.where(price_crossing_up_middle)[0]
        
        # Set signals
        for i in entry_indices:
            if i >= warmup:
                signals.iloc[i] = 1.0  # Long signal
                
        # Set flat signals at exit points
        for i in exit_indices:
            if i >= warmup:
                signals.iloc[i] = 0.0  # Flat signal
                
        return signals