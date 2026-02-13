from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mean_reversion_btc_bollinger_rsi_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=40, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.2),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        
        # Entry condition: price touches upper bollinger band and RSI > 70
        entry_long = (close == bb_upper) & (rsi > rsi_overbought)
        
        # Exit condition: price crosses below middle bollinger band
        exit_long = close < bb_middle
        
        # Initialize entry and exit points
        entry_points = pd.Series(False, index=df.index)
        exit_points = pd.Series(False, index=df.index)
        
        # Mark entry and exit points
        entry_points.loc[entry_long] = True
        exit_points.loc[exit_long] = True
        
        # Generate signals
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(len(df)):
            if not in_position and entry_points.iloc[i]:
                entry_price = close[i]
                stop_loss = entry_price - (stop_atr_mult * atr[i])
                take_profit = entry_price + (tp_atr_mult * atr[i])
                signals.iloc[i] = 1.0
                in_position = True
            elif in_position:
                if close[i] <= stop_loss or close[i] >= take_profit or exit_points.iloc[i]:
                    signals.iloc[i] = 0.0
                    in_position = False
                else:
                    signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
                
        return signals