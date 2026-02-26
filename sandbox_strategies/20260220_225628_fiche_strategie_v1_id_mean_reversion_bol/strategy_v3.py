from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
from utils.parameters import ParameterSpec

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
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64) # Initialize signal series
        n = len(df) # Number of data points
        warmup = int(params.get('warmup', 50)) # Warm-up period
        
        if "leverage" in params and params["leverage"] > 1: # Always enforce leverage <= 3 when provided.
            raise ValueError("Leverage must be <= 3")
            
        rsi = pd.Series(0, index=df.index) # Initialize RSI series
        
        if "rsi" not in indicators or "ema" not in indicators or "atr" not in indicators: 
            raise ValueError("RSI, EMA and ATR are required.")
            
        atr_mult = np.ones(len(df))*3 if "atr" not in indicators else (np.array(df["close"]) - df[['high', 'low']].max().values)/np.array(df[['high', 'low']].min() - df[['high', 'low']].mean()) # Calculate ATR based on close price and high/low prices
        
        ma_periods = [5, 10] if "ma" in indicators else [23, 45, 98] # No additional parameters required for this strategy.
        
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
            
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        if "leverage" in params and params["leverage"] > 1: # Always enforce leverage <= 3 when provided.
            raise ValueError("Leverage must be <= 3")
        
            
        signals.iloc[:warmup] = 0.0
    
    return signals