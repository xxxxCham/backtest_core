from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    """Stratégie générée par le builder."""

    def __init__(self):
        super().__init__(name="TestBuilder")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period": 14, "atr_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(
                name="rsi_period", min_val=5, max_val=30,
                default=14, param_type="int",
            ),
        }

    def generate_signals(
        self, df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = indicators.get("rsi")
        if rsi is not None:
            signals[rsi < 30] = 1.0
            signals[rsi > 70] = -1.0
        return signals
