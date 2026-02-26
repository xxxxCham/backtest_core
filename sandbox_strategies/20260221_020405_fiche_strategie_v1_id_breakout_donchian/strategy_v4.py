from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(df, default_params):
            # Define Bollinger Band parameters
            bollinger = ['bollinger', 'middle']  # e.g., ['donchian', 'upper'] for donchian bands

            # Initialize signals Series with zeros
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Iterate over each bar in the DataFrame
            for i, row in df.iterrows():
                close_price = row['close']  # Extract closing price from current bar

                # Define Bollinger Band
                donchian_band = row[bollinger + '_donchian'].values  

                # Donchian breakout (long position)
                prev_upper = np.roll(donchian_band, 1) if i > 0 else donchian_band   
                long_signal = close_price >= prev_upper[i]                            

                # Short term average True Range (ATR) based stop loss and take profit levels
                atr = row['atr']                                             
                atr_length = len(atr) - 1                                         
                short_ema = np.mean(row[bollinger + '_donchian'].values[-atr_length:])  
                long_ema = np.mean(row[bollinger + '_donchian'].values[:-atr_length]) # ema values are rounded to nearest integer for simplicity, adjust as needed

                short_tp = close_price + (long_ema - short_ema) * atr[-1]         
                long_slp = close_price - (short_ema - long_ema) * atr[-1]           

                # Long position conditions met 
                signals[i] += long_signal * 1.0 if long_tp > short_tp and indicators['adx']['adx'] > 30 else signals[i]  

                # Short position conditions met 
                signals[i] -= short_signal * -1.0 if short_slp < long_slp and indicators['adx']['adx'] > 30 else signals[i]   

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
