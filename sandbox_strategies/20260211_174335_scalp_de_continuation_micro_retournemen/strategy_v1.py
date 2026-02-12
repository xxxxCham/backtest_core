from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        
        # Get indicators dataframes
        ema_df = indicators['ema']
        rsi_df = indicators['rsi']
        bollinger_df = indicators['bollinger']
        
        # Ensure all dataframes are aligned with the main df
        ema_21 = ema_df['ema_21'].align(df['close'], join='left')[0]
        rsi = rsi_df['rsi_14'].align(df['close'], join='left')[0]
        lower_band = bollinger_df['lower'].align(df['close'], join='left')[0]
        upper_band = bollinger_df['upper'].align(df['close'], join='left')[0]
        
        # Calculate previous values for crossover detection
        prev_rsi = rsi.shift(1)
        
        # LONG signals
        long_conditions = (
            df['close'] > ema_21,
            (rsi > 50) & (prev_rsi <= 50),
            df['close'] > lower_band,
            df['close'].shift(1) < df['close'],
            df['close'] > lower_band.shift(1)
        )
        
        # SHORT signals
        short_conditions = (
            df['close'] < ema_21,
            (rsi < 50) & (prev_rsi >= 50),
            df['close'] < upper_band,
            df['close'].shift(1) > df['close'],
            df['close'] < upper_band.shift(1)
        )
        
        # Generate signals
        long_signals = pd.Series(0.0, index=df.index)
        short_signals = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            if all(long_conditions[j][i] for j in range(len(long_conditions))):
                long_signals.iloc[i] = 1.0
            if all(short_conditions[j][i] for j in range(len(short_conditions))):
                short_signals.iloc[i] = -1.0
        
        signals = long_signals + short_signals
        
        return signals