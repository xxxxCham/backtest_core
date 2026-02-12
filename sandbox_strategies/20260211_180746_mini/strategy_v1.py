from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mini")
    @property
    def required_indicators(self) -> List[str]:
        return ["rsi","bollinger","atr"]
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period":14,"rsi_oversold":30,"rsi_overbought":70,"stop_atr_mult":1.5,"tp_atr_mult":3.0,"warmup":50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5,max_val=50,default=14,param_type="int")}
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        close = df["close"].values
        signals[(rsi < params.get("rsi_oversold", 30)) & (close < lower)] = 1.0
        signals[(rsi > params.get("rsi_overbought", 70)) & (close > upper)] = -1.0
        signals.iloc[:50] = 0.0
        return signals