from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    """
    Dogecoin Mean Reversion Scalper
    Objective: Scalping strategy for DOGE using mean reversion at extreme RSI levels combined with Supertrend and ADX filters.
    Entry Conditions:
        LONG: price crosses above supertrend, RSI < 30, ADX < 25
        SHORT: price crosses below supertrend, RSI > 70, ADX < 25
    Exit Conditions:
        Trailing stop based on ATR (stop_loss_multiplier * ATR)
        Take profit at take_profit_multiplier * ATR
    Risk Management:
        Leverage limited to 1-2x
        Stop-loss range: 1.5-2.5× ATR
        Take-profit range: 2-4× ATR
    """

    def __init__(self):
        super().__init__(name="Dogecoin Mean Reversion Scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "supertrend", "adx"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "leverage": 1,
            "stop_loss_multiplier": 2,
            "take_profit_multiplier": 3,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "leverage": ParameterSpec(
                name="Leverage",
                type=float,
                min=1.0,
                max=2.0,
                default=self.default_params["leverage"],
            ),
            "stop_loss_multiplier": ParameterSpec(
                name="Stop Loss Multiplier",
                type=float,
                min=1.5,
                max=2.5,
                default=self.default_params["stop_loss_multiplier"],
            ),
            "take_profit_multiplier": ParameterSpec(
                name="Take Profit Multiplier",
                type=float,
                min=2.0,
                max=4.0,
                default=self.default_params["take_profit_multiplier"],
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicator values
        rsi = indicators["rsi"]
        supertrend = indicators["supertrend"]["supertrend"]
        adx = indicators["adx"]["adx"]
        atr = indicators.get("atr", np.zeros(n))  # Fallback to zeros if ATR not available

        # Initialize position tracking
        in_long = False
        in_short = False
        
        # Calculate entry conditions
        long_entry = (
            (df["close"] > supertrend) & 
            (rsi < 30) & 
            (adx < 25)
        )
        short_entry = (
            (df["close"] < supertrend) & 
            (rsi > 70) & 
            (adx < 25)
        )

        # Calculate exit conditions
        stop_loss_level = atr * params.get("stop_loss_multiplier", self.default_params["stop_loss_multiplier"])
        take_profit_level = atr * params.get("take_profit_multiplier", self.default_params["take_profit_multiplier"])

        for i in range(1, n):
            if not in_long and not in_short:
                # Enter LONG
                if long_entry[i] and np.nan_to_num(long_entry[i-1]) == 0:
                    signals[i] = 1.0
                    in_long = True
                    entry_price = df["close"][i]
                    
                # Enter SHORT
                elif short_entry[i] and np.nan_to_num(short_entry[i-1]) == 0:
                    signals[i] = -1.0
                    in_short = True
                    entry_price = df["close"][i]

            else:
                if in_long:
                    # Calculate trailing stop for LONG
                    stop_loss = entry_price - (stop_loss_level[i] * params.get("leverage", self.default_params["leverage"]))
                    take_profit = entry_price + (take_profit_level[i] * params.get("leverage", self.default_params["leverage"]))
                    
                    if df["close"][i] <= stop_loss:
                        signals[i] = 0.0
                        in_long = False
                    elif df["close"][i] >= take_profit:
                        signals[i] = 0.0
                        in_long = False

                elif in_short:
                    # Calculate trailing stop for SHORT
                    stop_loss = entry_price + (stop_loss_level[i] * params.get("leverage", self.default_params["leverage"]))
                    take_profit = entry_price - (take_profit_level[i] * params.get("leverage", self.default_params["leverage"]))
                    
                    if df["close"][i] >= stop_loss:
                        signals[i] = 0.0
                        in_short = False
                    elif df["close"][i] <= take_profit:
                        signals[i] = 0.0
                        in_short = False

        # Clean up NaN values
        signals = np.nan_to_num(signals)
        
        return pd.Series(signals, index=df.index, dtype=np.float64)