from utils.parameters import ParameterSpec
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['macd', 'roc', 'atr', 'bollinger']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Extract indicator arrays
        macd_line = indicators['macd']['macd']
        signal_line = indicators['macd']['signal']
        roc = indicators["roc"]
        atr = indicators["atr"]
        bb_upper = indicators['bollinger']['upper']
        bb_middle = indicators['bollinger']['middle']
        bb_lower = indicators['bollinger']['lower']
        
        # Calculate roc_diff as difference of ROC
        roc_diff = np.zeros_like(roc)
        roc_diff[1:] = np.diff(roc)
        
        # Calculate atr_mean (simple moving average of atr)
        atr_mean = np.zeros_like(atr)
        window = 20
        for i in range(len(atr)):
            if i < window:
                atr_mean[i] = np.mean(atr[:i+1])
            else:
                atr_mean[i] = np.mean(atr[i-window+1:i+1])
        
        # Create boolean masks for long condition
        long_condition = (
            (macd_line > signal_line) &
            (roc > 0) &
            (roc_diff > 0) &
            (df['close'] > bb_upper) &
            (atr > atr_mean)
        )
        
        # Create boolean masks for short condition
        short_condition = (
            (macd_line < signal_line) &
            (roc < 0) &
            (roc_diff < 0) &
            (df['close'] < bb_lower) &
            (atr > atr_mean)
        )
        
        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0
        
        return signals