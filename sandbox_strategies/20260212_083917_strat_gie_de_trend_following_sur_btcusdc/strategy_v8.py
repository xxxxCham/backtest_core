from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BTCUSDC_30m_Trend_Following")

    @property
    def required_indicators(self) -> List[str]:
        return ["sma", "adx", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"adx_period": 14, "atr_period": 14, "sma_period": 20, "stop_atr_mult": 2.0, "tp_atr_mult": 5.0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "adx_period": ParameterSpec(param_name="adx_period", param_type="int", min_value=5, max_value=30, step=1),
            "atr_period": ParameterSpec(param_name="atr_period", param_type="int", min_value=5, max_value=30, step=1),
            "sma_period": ParameterSpec(param_name="sma_period", param_type="int", min_value=10, max_value=50, step=1),
            "stop_atr_mult": ParameterSpec(param_name="stop_atr_mult", param_type="float", min_value=1.0, max_value=5.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_name="tp_atr_mult", param_type="float", min_value=2.0, max_value=10.0, step=0.5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        adx_period = params["adx_period"]
        atr_period = params["atr_period"]
        sma_period = params["sma_period"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        
        close = np.nan_to_num(df["close"].values)
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions: price crosses below SMA and ADX > 20
        price_cross_below_sma = (close < sma) & (np.roll(close, 1) >= np.roll(sma, 1))
        adx_filter = adx > 20
        
        # Exit conditions: price crosses above SMA or ADX < 15
        price_cross_above_sma = (close > sma) & (np.roll(close, 1) <= np.roll(sma, 1))
        adx_divergence = adx < 15
        
        # Generate short signals
        entry_mask = price_cross_below_sma & adx_filter
        exit_mask = price_cross_above_sma | adx_divergence
        
        # Initialize signal array
        signal_values = np.zeros_like(close)
        
        # Set initial short signals
        signal_values[entry_mask] = -1.0
        
        # Track open positions
        in_position = np.zeros_like(close, dtype=bool)
        position_entry_price = np.full_like(close, np.nan)
        stop_loss = np.full_like(close, np.nan)
        take_profit = np.full_like(close, np.nan)
        
        for i in range(1, len(close)):
            # If entering a position
            if signal_values[i] == -1.0:
                in_position[i] = True
                position_entry_price[i] = close[i]
                stop_loss[i] = close[i] - (atr[i] * stop_atr_mult)
                take_profit[i] = close[i] + (atr[i] * tp_atr_mult)
            elif in_position[i-1] and not in_position[i]:
                # Check for exit conditions
                if close[i] >= take_profit[i-1] or close[i] <= stop_loss[i-1]:
                    signal_values[i] = 0.0
                    in_position[i] = False
                elif exit_mask[i]:
                    signal_values[i] = 0.0
                    in_position[i] = False
                else:
                    signal_values[i] = -1.0
                    in_position[i] = True
            else:
                # No change in position
                signal_values[i] = signal_values[i-1]
        
        # Convert to Series
        signals = pd.Series(signal_values, index=df.index, dtype=np.float64)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals