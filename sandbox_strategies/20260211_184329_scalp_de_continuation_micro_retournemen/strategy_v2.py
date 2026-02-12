from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BuilderStrategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "period": ParameterSpec(
                type_=int,
                description="Period for RSI calculation.",
                constraints=(2, 50),
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        ema_21 = np.nan_to_num(indicators["ema"][("close", 21)])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        # logic here
        return signals