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
            "adx_period": ParameterSpec(10, 30, 1),
            "atr_period": ParameterSpec(10, 30, 1),
            "sma_period": ParameterSpec(10, 50, 1),
            "stop_atr_mult": ParameterSpec(1.0, 5.0, 0.5),
            "tp_atr_mult": ParameterSpec(2.0, 10.0, 0.5),
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
        
        sma = np.nan_to_num(indicators["sma"])
        adx = np.nan_to_num(indicators["adx"]["adx"])
        atr = np.nan_to_num(indicators["atr"])
        
        # Entry conditions for short trades
        price = df["close"].values
        price_above_sma = price > sma
        adx_strong = adx > 20
        
        # Exit condition when price crosses SMA and ADX weakens
        price_below_sma = price < sma
        adx_weak = adx < 20
        
        # Short entry signal
        short_entry = price_above_sma & adx_strong
        
        # Short exit signal
        short_exit = price_below_sma & adx_weak
        
        # Generate signals
        entry_indices = np.where(short_entry)[0]
        exit_indices = np.where(short_exit)[0]
        
        # Set signals for short entries
        for i in entry_indices:
            if i > 0:
                signals.iloc[i] = -1.0
                
        # Set signals for exits
        for i in exit_indices:
            if i > 0 and signals.iloc[i-1] == -1.0:
                signals.iloc[i] = 0.0
                
        # Apply stop-loss and take-profit logic
        entry_price = np.full(len(df), np.nan)
        entry_indices = np.where(signals == -1.0)[0]
        for i in entry_indices:
            entry_price[i] = price[i]
            
        # Check for stop-loss and take-profit
        for i in range(len(df)):
            if not np.isnan(entry_price[i]) and signals.iloc[i] == -1.0:
                sl = entry_price[i] + (stop_atr_mult * atr[i])
                tp = entry_price[i] - (tp_atr_mult * atr[i])
                if price[i] >= sl:
                    signals.iloc[i] = 0.0
                elif price[i] <= tp:
                    signals.iloc[i] = 0.0
                    
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        
        return signals