from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='phase_lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 25,
         'bollinger_stddev': 2,
         'leverage': 1,
         'momentum_nperiods': 10,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_nperiods': ParameterSpec(
                name='momentum_nperiods',
                min_val=1,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'bollinger_stddev': ParameterSpec(
                name='bollinger_stddev',
                min_val=1,
                max_val=20,
                default=2,
                param_type='float',
                step=0.1,
            ),
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
        def generate_signals(df):
            # Define default parameters
            default_params = {
                'leverage': 1.,  
                'close':'last',  
                'sl_tp':0,      
                'bb_period' :20     
            }

            # Create a copy of dataframe as changes are made in-place
            df_mod = df.copy(deep=True) 

            signals = pd.Series(np.nan, index=df.index, dtype=np.float64)  

            for i, row in df_mod.iterrows():    
                # Indicators available in this method: ['momentum', 'bollinger', 'atr']

                # LONG intent: momentum >= x AND indicators['bollinger']['middle'] <= close
                if np.max(row['momentum'])>=0.7 and row['bollinger']['middle'].values<=row['close']:  
                    signals[i] = 1.0   

                # SHORT intent: momentum < y AND indicators['bollinger']['lower'] <= close OR momentum > z AND indicators['bollinger']['upper'] <= close
                elif np.min(row['momentum'])<0.3 and row['bollinger']['lower'].values<=row['close']:  
                    signals[i] = -1.0   

                # ADI: check the direction of the trend 
                elif abs(np.mean((row["np.nan_to_num(indicators['donchian']['upper'])"][:-1]-row["np.nan_to_num(indicators['donchian']['upper'])"][1:]))/2.) > row['adx']['minus_di'][i]:  
                    signals[i] = -1.0   

                 # Supertrend
                elif np.mean((row["np.nan_to_num(indicators['supertrend']['direction'])"]["down"][:-1]-row["np.nan_to_num(indicators['supertrend']['direction'])"]["up"][1:])) > 2*np.std(row["np.nan_to_num(indicators['supertrend']['direction'])"]["up"]):  
                    signals[i] = -1.0   

                 # Stochastic
                elif (row['stochastic']['stoch_k'][i]<=30 and row['stochastic']['stoch_d'][i]>70) or \
                      (row['stochastic']['stoch_k'][i]>70 and row['stochastic']['stoch_d'][i]<=30):  
                    signals[i] = -1.0   

                # ATR: check if the price is outside of Bollinger Band 
                elif abs(row["close"]-row["bb_stop_long"][i]) > row['bb_period']:    
                    signals[i] = 1.0  

                else:   
                     signals[i]=0.0     

            signals=signals # Assign the generated signal to a new column in dataframe named "signal"
        signals.iloc[:warmup] = 0.0
        return signals
