from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vol_amplitude_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(indicators, df):
            # Get Bollinger Bands information
            bollinger = indicators['bollinger']

            # Get Donchian Channels information
            donchian = indicators['donchian']

            # ADX calculation 
            adx = indicators['adx']

            # SuperTrend calculation 
            super_trend = indicators['supertrend']

            # Stochastic calculation  
            stochastic = indicators['stochastic']

            # ATR Calculation 
            atr = indicators.get('atr')

            for i, row in df.iterrows():
                close_price = np.array(row[df['close']])

                # Close price breaks above Bollinger Bands Upper and Donchian Channel Upper 
                if indicators['bollinger']["upper"][i] > close_price[-1]:
                    signals.append(1)

                # Close price breaks below Bollinger Bands Lower and Donchian Channel Lower  
                elif indicators['bollinger']["lower"][i] < close_price[-1]:
                    signals.append(-1) 

                # ADX calculation based on previous high, low, and close prices   
                adx[i] = _adx(row[[df['high'], df['low'], row[df['close']]]])

                # SuperTrend crossing of the price over its direction  
                if super_trend[i][0] > super_trend[i-1]:
                    signals.append(1) 

                # Stochastic crossover   
                if indicators['stochastic']["stoch_k"][i] < indicators['stochastic']["stoch_d"][i] and close_price[-1] > close_price[-2]*adx[i]['deverage']:
                    signals.append(-1)  

                elif (indicators['stochastic']["stoch_k"][i] >= indicators['stochastic']["stoch_d"][i]) and close_price[-1] < close_price[-2]*adx[i]['deverage']: 
                    signals.append(1)  

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
