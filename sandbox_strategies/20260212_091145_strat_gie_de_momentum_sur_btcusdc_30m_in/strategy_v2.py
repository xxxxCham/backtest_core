from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="momentum_btcusdc_30m_revised")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(10, 30, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 1)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]
        
        # Params
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        warmup = int(params.get("warmup", 50))
        
        # Entry conditions
        entry_long = (rsi > rsi_overbought) & (close < bb_lower) & (prev_close >= bb_lower) & (rsi_prev <= rsi_overbought)
        entry_short = (rsi < rsi_oversold) & (close > bb_upper) & (prev_close <= bb_upper) & (rsi_prev >= rsi_oversold)
        
        # Signal generation
        long_signal = np.zeros_like(rsi)
        short_signal = np.zeros_like(rsi)
        
        # Set initial signals
        long_signal[entry_long] = 1.0
        short_signal[entry_short] = -1.0
        
        # Combine signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0
        
        # Create mask for active positions
        long_positions = np.zeros_like(rsi, dtype=bool)
        short_positions = np.zeros_like(rsi, dtype=bool)
        entry_prices = np.zeros_like(rsi)
        
        for i in range(1, len(rsi)):
            # Close existing positions
            if long_positions[i-1]:
                # Exit if RSI overbought or stop-loss/take-profit hit
                if (rsi[i] > rsi_overbought) or (close[i] > entry_prices[i-1] + tp_atr_mult * atr[i-1]) or (close[i] < entry_prices[i-1] - stop_atr_mult * atr[i-1]):
                    signals.iloc[i] = 0.0
                    long_positions[i] = False
                else:
                    signals.iloc[i] = 1.0
            elif short_positions[i-1]:
                # Exit if RSI oversold or stop-loss/take-profit hit
                if (rsi[i] < rsi_oversold) or (close[i] < entry_prices[i-1] - tp_atr_mult * atr[i-1]) or (close[i] > entry_prices[i-1] + stop_atr_mult * atr[i-1]):
                    signals.iloc[i] = 0.0
                    short_positions[i] = False
                else:
                    signals.iloc[i] = -1.0
            else:
                # Check for new entries
                if long_signal[i]:
                    signals.iloc[i] = 1.0
                    long_positions[i] = True
                    entry_prices[i] = close[i]
                elif short_signal[i]:
                    signals.iloc[i] = -1.0
                    short_positions[i] = True
                    entry_prices[i] = close[i]
        
        # Set warmup period
        signals.iloc[:warmup] = 0.0
        
        return signals