import numpy as np
from typing import Any, Dict, List

class BuilderGeneratedStrategy:
    def __init__(self):
        self._params = None # Params are defined in the ParameterSpec object and passed to StrategyBase.__init__() method.
        self._indicators = {}  # Indicators dictionary is pre-computed by engine and accessed via `get_indicators()` function
    
    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'rsi']   # This list will be dynamically generated according to what indicators are provided.
        
    @property
    def default_params(self) -> Dict[str, Any]:
        return {'warmup': 50}     # Default parameters.

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        # This method is used to define the specifications for each tunable parameter in your strategy.
        return {  
            'warmup' : ParameterSpec('Warm Up Period', type=int, default_value=50), 
        }
        
    @staticmethod
    def get_indicators() -> Dict[str, Any]:
        # This method is used to dynamically generate indicators dictionary based on what indicators are provided.
        return {'bollinger': {}, 'atr': {}, 'rsi': {}}  
    
    def _get_indicator(self, name: str) -> np.ndarray:  # Helper function for retrieving a single indicator array from the indicators dict.
        if self._indicators is None or not isinstance(name, str): return None
        return self._indicators[name]
    
    def _get_indicator_values(self, name: str) -> Any:  # Helper function for retrieving a single indicator value from the indicators dict.
        if self._indicators is None or not isinstance(name, str): return None
        return self._indicators[name]
    
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any], indicators:Dict[str, np.ndarray]) -> pd.Series:  # Main logic for generating signals to go long or short.
        
        signals = pd.Series([0.0]*len(df), index=df.index)   # Initialize Series with all zeros.
            
        if self._params is None or not isinstance(params, dict): return signals  # Check if parameters are defined and passed correctly.
    
        warmup_period = params['warmup']    # Get Warm Up Period from parameters.
        
        for name in self.required_indicators:   # Loop over required indicators to generate signals based on their conditions.
            indicator = indicators[name]      
            
            if 'upper' in name or 'lower' in name:  # If upper or lower bound is present, compare the current price with that level and set signal accordingly.
                threshold_value = self._get_indicator(f"{name}_threshold")  
                
                if not isinstance(threshold_value, np.ndarray): return signals    # Check if threshold value exists for this indicator or not.
                    
                current_price = df['close'][0]  # Assuming close price is the first data point in the dataframe.
            
                if '>' in name:   # If comparison operator is greater than, set signal to long when price goes below threshold and short otherwise.
                    signals[current_price < indicator] = -1     # Set Signal to Long (Negative Sign) 
                    
            elif name == 'rsi':    # RSI uses current and average price, so we subtract the former from the latter to get delta values.
                rsi = indicators['rsi']        
                
                if not isinstance(rsi, np.ndarray): return signals   # Check if RSI value exists for this indicator or not.
            
            elif name == 'atr':    # ATR uses standard deviation of price over a specific period to calculate the true range and then applies it to current high-low values.
                atr = indicators['atr']        
                
                if not isinstance(atr, np.ndarray): return signals   # Check if ATR value exists for this indicator or not.
            
        # Remove first warmup_period bars from signal series as they are not part of the real trading period.
        signals = signals[warmup_period:] 
        
        return signals    # Return final generated signals after all logic is applied.