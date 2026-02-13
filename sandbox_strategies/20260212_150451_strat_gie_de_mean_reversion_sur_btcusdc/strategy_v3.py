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
        return ["donchian", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(50, 90, 1),
            "rsi_oversold": ParameterSpec(10, 50, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(0.5, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(1.0, 4.0, 0.1),
            "warmup": ParameterSpec(10, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        donchian = indicators["donchian"]
        rsi = indicators["rsi"]
        atr = indicators["atr"]
        
        # Get donchian values
        upper_band = np.nan_to_num(donchian["upper"])
        middle_band = np.nan_to_num(donchian["middle"])
        lower_band = np.nan_to_num(donchian["lower"])
        
        # Get RSI values
        rsi = np.nan_to_num(rsi)
        
        # Get ATR values
        atr = np.nan_to_num(atr)
        
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        # Entry conditions
        # Price touches upper Donchian band
        price_touches_upper = np.isclose(df["close"].values, upper_band, rtol=1e-5)
        
        # RSI in overbought territory
        rsi_overbought = rsi > params["rsi_overbought"]
        
        # Price above upper Bollinger band (using bollinger from indicators)
        bb = indicators.get("bollinger")
        if bb is not None:
            bb_upper = np.nan_to_num(bb["upper"])
            price_above_bb_upper = df["close"].values > bb_upper
        else:
            price_above_bb_upper = np.full(len(df), False)
        
        # ADX less than 25
        adx = indicators.get("adx")
        if adx is not None:
            adx_value = np.nan_to_num(adx["adx"])
            adx_below_threshold = adx_value < 25
        else:
            adx_below_threshold = np.full(len(df), True)
        
        # Entry signal
        entry_signal = price_touches_upper & rsi_overbought & price_above_bb_upper & adx_below_threshold
        
        # Exit conditions
        # Price crosses below middle Donchian band
        price_crosses_below_middle = df["close"].values < middle_band
        
        # Price crosses below lower Bollinger band
        if bb is not None:
            bb_lower = np.nan_to_num(bb["lower"])
            price_crosses_below_bb_lower = df["close"].values < bb_lower
        else:
            price_crosses_below_bb_lower = np.full(len(df), False)
        
        # Exit signal
        exit_signal = price_crosses_below_middle | price_crosses_below_bb_lower
        
        # Set signals
        entry_indices = np.where(entry_signal)[0]
        exit_indices = np.where(exit_signal)[0]
        
        for i in entry_indices:
            if i < len(signals):
                signals.iloc[i] = 1.0  # Long signal
                
                # Apply stop-loss and take-profit logic
                stop_loss_atr = atr[i] * params["stop_atr_mult"]
                take_profit_atr = atr[i] * params["tp_atr_mult"]
                
                # Find next exit point
                for j in range(i+1, len(signals)):
                    if exit_signal[j]:
                        signals.iloc[j] = 0.0  # Flat
                        break
                        
        return signals