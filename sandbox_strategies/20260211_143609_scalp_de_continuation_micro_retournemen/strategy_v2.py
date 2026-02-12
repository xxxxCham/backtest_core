from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ScalpContinuationBands
    # Objective: Scalp de continuation/micro-retournement on liquid crypto (BTCUSDC)
    # Timeframe: 30m
    
    def __init__(self):
        super().__init__(name="ScalpContinuationBands")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "ema", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_period_short": 21,
            "ema_period_long": 50,
            "rsi_period": 14,
            "risk_percentage": 1.5,
            "stop_loss_type": "EMA",
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(int, default=20, min=10, max=30),
            "ema_period_short": ParameterSpec(int, default=21, min=10, max=50),
            "ema_period_long": ParameterSpec(int, default=50, min=30, max=100),
            "rsi_period": ParameterSpec(int, default=14, min=7, max=28),
            "risk_percentage": ParameterSpec(float, default=1.5, min=1.0, max=2.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required data
        close = df["close"].values
        ema_short = np.nan_to_num(indicators["ema"])
        ema_long = np.nan_to_num(indicators["ema"])  # EMA50
        rsi = np.nan_to_num(indicators["rsi"])
        
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])

        # Calculate EMA21 and EMA50 crossover conditions
        ema_short_condition = close > ema_short
        ema_long_condition = close > ema_long

        # RSI conditions for trend continuation
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # Bollinger band rejection conditions
        lower_bb_reject = (close[-2] < lower_bb[-2]) & (close[-1] > close[-2])
        upper_bb_reject = (close[-2] > upper_bb[-2]) & (close[-1] < close[-2])

        # Generate LONG entries
        long_entry = (
            ema_short_condition &
            ~ema_long_condition &
            rsi_oversold &
            lower_bb_reject
        )

        # Generate SHORT entries
        short_entry = (
            ~ema_short_condition &
            ema_long_condition &
            rsi_overbought &
            upper_bb_reject
        )

        # Set signals for valid entries (starting from index 50 to ensure data)
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Exit conditions: opposite band or divergence
        exit_long = close > upper_bb
        exit_short = close < lower_bb
        
        # Apply exits by setting signals back to 0
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        return signals