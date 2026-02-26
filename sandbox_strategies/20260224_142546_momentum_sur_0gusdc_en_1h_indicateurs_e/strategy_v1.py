from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')
        
    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(name='leverage', min_val=1, max_val=2, default=1, param_type='int', step=1),
            'stop_atr_mult': ParameterSpec(name='stop_atr_mult', min_val=1.0, max_val=2.0, default=1.5, param_type='float', step=0.1),
            'tp_atr_mult': ParameterSpec(name='tp_atr_mult', min_val=2.0, max_val=4.5, default=3.0, param_type='float', step=0.1),
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        long_bollinger, short_donchian = zip(*indicators['bollinger'].items())  # Dealing with sub-keys in indicators dictionary
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        for i, (long_bollinger, short_donchian) in enumerate(zip(indicators['bollinger'], indicators['donchian'])):
            close = df['close'][i]  # The closing price of the current bar.
            
            long_upper, long_middle, long_lower = np.array([long_bollinger[j]] for j in range(n)) if isinstance(long_bollinger, dict) else (np.zeros((1,)),) + long_bollinger  # The three Bollinger Bands for the current bar.
            short_upper, short_middle, short_lower = np.array([short_donchian[j]] for j in range(n)) if isinstance(short_donchian, dict) else (np.zeros((1,)),) + short_donchian  # The three Bollinger Bands for the current bar.
            
            prev_long_upper = long_upper[:-2] if i > 0 else np.roll(long_upper, -1)[:-2]  # Previous upper band of previous bar.
            prev_long_middle = long_middle[:-2] if i > 0 else np.roll(long_middle, -1)[:-2]  # Previous middle band of previous bar.
            prev_long_lower = long_lower[:-2] if i > 0 else np.roll(long_lower, -1)[:-2]  # Previous lower band of previous bar.
            
            prev_short_upper = short_upper[:-2] if i > 0 else np.roll(short_upper, -1)[:-2]  # Previous upper band of previous bar.
            prev_short_middle = short_middle[:-2] if i > 0 else np.roll(short_middle, -1)[:-2]  # Previous middle band of previous bar.
            prev_short_lower = short_lower[:-2] if i > 0 else np.roll(short_lower, -1)[:-2]  # Previous lower band of previous bar.
            
            upper_band_crossed = close >= long_upper[i] and close < prev_long_upper[-1]
            donchian_broken = (close <= short_middle[i]) or (close > short_lower[i])
            adx_above_25 = indicators['adx'][i, 0] >= 25 if 'adx' in indicators else False
            
            # Generate long signal.
            if upper_band_crossed and donchian_broken:
                signals[i] = 1.0
                
        for i, (long_bollinger, short_donchian) in enumerate(zip(indicators['bollinger'], indicators['donchian'])):
            close = df['close'][i]   # The closing price of the current bar.
            
            upper, middle, lower = np.array([short_bollinger[j]] for j in range(n)) if isinstance(short_bollinger, dict) else (np.zeros((1,)),) + short_bollinger  # The three Bollinger Bands for the current bar.
            prev_upper = upper[:-2] if i > 0 else np.roll(upper, -1)[:-2]   # Previous upper band of previous bar.
            prev_middle = middle[:-2] if i > 0 else np.roll(middle, -1)[:-2]     # Previous middle band of previous bar.
            prev_lower = lower[:-2] if i > 0 else np.roll(lower, -1)[:-2]         # Previous lowter band of previous bar.
            
            upper_band_crossed = close >= prev_upper[-1] and close < prev_middle[i]   # The two Bollinger Bands for the current bar.
            donchian_broken = (close <= prev_lower[i]) or (close > prev_middle[i])  # Donchian breakout rule for the current bar.
            adx_above_25 = indicators['adx'][i, 0] >= 25 if 'adx' in indicators else False   # ADX above 25.
            
            # Generate long signal.
            if upper_band_crossed and donchian_broken:
                signals[i] = -1.0
                
        signals[:warmup] = 0.0
        
    return signals