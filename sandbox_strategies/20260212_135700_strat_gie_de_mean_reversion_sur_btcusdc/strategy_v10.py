from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="keltner_cci_mean_reversion_v10")

    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_period": 14, "keltner_multiplier": 1.5, "keltner_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "cci_period": ParameterSpec(param_name="cci_period", param_type="int", min_value=5, max_value=30, step=1),
            "keltner_multiplier": ParameterSpec(param_name="keltner_multiplier", param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "keltner_period": ParameterSpec(param_name="keltner_period", param_type="int", min_value=10, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=3.0, max_value=10.0, step=0.5),
            "warmup": ParameterSpec(param_name="warmup", param_type="int", min_value=20, max_value=100, step=10),
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
        keltner_middle = np.nan_to_num(keltner["middle"])
        keltner_lower = np.nan_to_num(keltner["lower"])
        
        # CCI lagged
        cci_lag1 = np.roll(cci, 1)
        cci_lag2 = np.roll(cci, 2)
        
        # Entry condition: price breaks out of lower Keltner band with CCI showing mean-reversion
        entry_long = (df['close'] > keltner_lower) & (cci < -100) & (cci_lag1 > -100) & (cci_lag2 > cci_lag1)
        
        # Exit condition: price crosses back to middle Keltner band
        exit_long = df['close'] > keltner_middle
        
        # Generate signals
        entry_indices = np.where(entry_long)[0]
        exit_indices = np.where(exit_long)[0]
        
        for i in entry_indices:
            if i < len(signals):
                signals.iloc[i] = 1.0  # LONG signal
                
        # Set exit signals
        for i in exit_indices:
            if i < len(signals) and signals.iloc[i] == 1.0:
                signals.iloc[i] = 0.0  # FLAT signal
                
        return signals