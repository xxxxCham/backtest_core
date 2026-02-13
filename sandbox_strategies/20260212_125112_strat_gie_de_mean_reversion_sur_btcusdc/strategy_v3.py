from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
from utils.parameters import ParameterSpec
from indicators import BollingerBand, AverageTrueRange, CommodityChannelIndex

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="Mean-reversion with Keltner + ATR+CCI")
        self.required_indicators = ["KELTNER", "ATR", "CCI"]
        self.default_params = {"ATR_period": 14, "CCI_overbought": 100, "CCI_oversold": -100, "KELTNER_overbought": 80, "KELTNER_oversold": 20}
        self.parameter_specs = {
            # fill each tunable parameter here with ParameterSpec objects
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic here
        
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals