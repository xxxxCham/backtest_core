from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btcusdc_30m_trend_following_adx_sma")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "atr_period": 14, "bb_period": 20, "bb_std_dev": 2, "sma_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "atr_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "bb_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
            "bb_std_dev": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "sma_period": ParameterSpec(param_type="int", min_value=10, max_value=50, step=1),
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
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        entry_short = (close < sma) & (close < bb_lower) & (adx > 25)
        
        # Exit conditions
        exit_short = (close > sma) | (adx < 15)
        
        # Generate signals
        positions = pd.Series(0.0, index=df.index)
        in_position = False
        entry_price = 0.0
        
        for i in range(len(df)):
            if not in_position and entry_short[i]:
                positions.iloc[i] = -1.0
                in_position = True
                entry_price = close[i]
            elif in_position and exit_short[i]:
                positions.iloc[i] = 0.0
                in_position = False
            elif in_position:
                positions.iloc[i] = -1.0
        
        signals = positions
        return signals