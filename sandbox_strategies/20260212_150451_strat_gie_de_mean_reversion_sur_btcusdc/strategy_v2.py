from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="donchian_rsi_mean_reversion")

    @property
    def required_indicators(self) -> List[str]:
        return ["donchian", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 30, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1),
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
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        donchian = indicators["donchian"]
        bb = indicators["bollinger"]
        
        # Donchian bands
        upper_donchian = np.nan_to_num(donchian["upper"])
        middle_donchian = np.nan_to_num(donchian["middle"])
        lower_donchian = np.nan_to_num(donchian["lower"])
        
        # Bollinger bands
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        
        # Price
        close = np.nan_to_num(df["close"].values)
        
        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 70)
        price_touches_upper_donchian = np.isclose(close, upper_donchian, rtol=1e-5)
        rsi_condition = rsi > rsi_overbought
        price_above_upper_bb = close > upper_bb
        
        # Filter out strong uptrends (ADX < 25)
        adx = np.nan_to_num(indicators["adx"]["adx"])
        adx_filter = adx < 25
        
        # Entry signal
        entry_long = price_touches_upper_donchian & rsi_condition & price_above_upper_bb & adx_filter
        
        # Exit conditions
        price_crosses_below_middle_donchian = close < middle_donchian
        price_crosses_below_lower_bb = close < lower_bb
        exit_signal = price_crosses_below_middle_donchian | price_crosses_below_lower_bb
        
        # Set signals
        entry_indices = np.where(entry_long)[0]
        for i in entry_indices:
            if i + 1 < len(signals):
                signals.iloc[i] = 1.0  # Long signal
                # Apply stop-loss and take-profit
                stop_loss_atr = atr[i] * params.get("stop_atr_mult", 1.5)
                take_profit_atr = atr[i] * params.get("tp_atr_mult", 3.0)
                # Simple exit logic based on exit signal
                for j in range(i+1, len(signals)):
                    if exit_signal[j]:
                        signals.iloc[j] = 0.0  # Flat
                        break
                        
        return signals