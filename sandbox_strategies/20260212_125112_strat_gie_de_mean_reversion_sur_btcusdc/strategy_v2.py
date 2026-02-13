from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="my_strat")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # fill each tunable parameter
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        if "change_type" in params and params["change_type"] == "logic":
            self._apply_proposed_changes(params, df, indicators)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals
    
    def _apply_proposed_changes(self, params: Dict[str, Any], df: pd.DataFrame, indicators: Dict[str, Any]):
        change_type = params["change_type"]
        if change_type == "params":
            self._modify_parameters(params)
        elif change_type == "logic":
            self._apply_proposed_changes_in_logic(df, indicators)
            
    def _modify_parameters(self, params: Dict[str, Any]):
        # modify parameters here. Replace these `pass` with actual logic
        pass 
    
    def _apply_proposed_changes_in_logic(self, df: pd.DataFrame, indicators: Dict[str, Any]):
        raise NotImplementedError("Proposed change type is not recognized.")