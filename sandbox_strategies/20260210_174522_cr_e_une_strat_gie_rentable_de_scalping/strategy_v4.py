from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Dogecoin Mean-Reversion Scalper strategy implementation
    
    def __init__(self):
        super().__init__(name="Dogecoin Mean-Reversion Scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "adx_period": 14,
            "atr_period": 14,
            "bollinger_period": 20,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "adx_threshold": 25,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "stop_atr_mult": ParameterSpec(
                name="Stop-loss ATR multiplier",
                type=float,
                min=1.5,
                max=2.5,
                default=1.5
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take-profit ATR multiplier",
                type=float,
                min=2.0,
                max=4.0,
                default=3.0
            ),
            "adx_threshold": ParameterSpec(
                name="ADX trend strength threshold",
                type=int,
                min=20,
                max=30,
                default=25
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI oversold level",
                type=int,
                min=20,
                max=40,
                default=30
            ),
            "rsi_overbought": ParameterSpec(
                name="RSI overbought level",
                type=int,
                min=60,
                max=80,
                default=70
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required indicator data
        close = df["close"].values
        
        rsi_val = np.nan_to_num(indicators["rsi"])
        
        bollinger = indicators["bollinger"]
        upper_bb = np.nan_to_num(bollinger["upper"])
        lower_bb = np.nan_to_num(bollinger["lower"])
        
        adx_data = indicators["adx"]
        adx_val = np.nan_to_num(adx_data["adx"])
        
        atr_val = np.nan_to_num(indicators["atr"])

        # Calculate thresholds
        stop_loss = atr_val * params.get("stop_atr_mult", 1.5)
        take_profit = atr_val * params.get("tp_atr_mult", 3.0)
        adx_filter = params.get("adx_threshold", 25)

        for i in range(1, n):
            # Skip if not enough data
            if i < max(params.get("bollinger_period", 20), 
                      params.get("rsi_period", 14)):
                continue

            current_close = close[i]
            prev_close = close[i-1]

            # Entry conditions
            long_entry = (
                (prev_close <= upper_bb[i-1]) & 
                (current_close > upper_bb[i]) &
                (rsi_val[i] < params.get("rsi_oversold", 30)) &
                (adx_val[i] >= adx_filter)
            )
            
            short_entry = (
                (prev_close >= lower_bb[i-1]) &
                (current_close < lower_bb[i]) &
                (rsi_val[i] > params.get("rsi_overbought", 70)) &
                (adx_val[i] >= adx_filter)
            )

            # Exit conditions
            long_exit = (
                current_close <= lower_bb[i] or 
                (current_close - close[i-1]) < -(stop_loss[i])
            )
            
            short_exit = (
                current_close >= upper_bb[i] or 
                (close[i-1] - current_close) < -(stop_loss[i])
            )

            # Apply signals
            if long_entry:
                signals.iloc[i] = 1.0
            elif short_entry:
                signals.iloc[i] = -1.0
            elif signals.iloc[i-1] == 1.0 and (long_exit or i >= take_profit[i]):
                signals.iloc[i] = 0.0
            elif signals.iloc[i-1] == -1.0 and (short_exit or i >= take_profit[i]):
                signals.iloc[i] = 0.0

        return signals