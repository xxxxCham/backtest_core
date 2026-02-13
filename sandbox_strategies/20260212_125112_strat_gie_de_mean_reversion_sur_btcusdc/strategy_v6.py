from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="Mean-reversion on BTCUSDC 30m using Keltner + CCI + ATR")
        
    @property
    def required_indicators(self) -> List[str]:
        return ["cci"]
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_mult": 1, "bollinger_stddev": 2, "cci_multiplier": 0.01, "rsi_period": 14}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # Fill each tunable parameter here. It's important to follow the correct data types and formats for parameters like `int`, `float`, `bool`, etc. 
        }
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Implement explicit LONG / SHORT / FLAT logic here. 
        # For example:
        # if indicators["cci"].signal > -15 and close < lower_bollingerband: long(signals = 1.0);
                
        warmup = int(params.get("warmup", 50))
        
        signals[:warmup] = 0.0   # Warm-up protection
        
        return signals