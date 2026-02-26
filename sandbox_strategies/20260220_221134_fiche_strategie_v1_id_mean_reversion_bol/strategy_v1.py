from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Mean Reversion Bollinger Rsi Attr High Low StopLossAtraTakeProfitExitCondition CrossAny Close Middle AtrMiddle BollingerUpper')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 4.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val='2.5',
                max_val='4.5',
                default='2.5',
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=4.5,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
                signals = pd.Series(0.0, index=df.index)

                # Convert Bollinger Bands and RSI data into numpy arrays for computations
                close = df["close"].values
                upper, middle, lower = indicators['bollinger'][:-1]
                upper_band = upper - (upper - middle) * params['atr']['atr_mult'] 
                middle_band = middle  
                lower_band = lower + (lower - middle) * params['atr']['atr_mult']   

                bollinger_bands = pd.concat([lower_band - upper_band, middle_band, upper_band + upper_band], axis=1) 
                rsi = indicators['rsi'].values  
                atr = indicators['atr'].values

                long_mask = np.zeros(len(df), dtype=bool)

                # Implement your logic for entering a long position here.
                if rsi > params['threshold']: 
                    long_mask[params["long_from"]:] = True  

                # Implement your logic for exiting the position here.
                elif rsi < params['exit_rsi'] and len(df) >= params['trailing_profit']['sl_start']:   
                    signals[(signals>0)&((close>=lower[params["long_from"]])& (close<bollinger_bands[:-1][:, -1]))] = 1.0  

                # Implement your logic for exiting the position if ATR based stop loss is hit.
                elif params['trailing_profit']['atr']:   
                    signals[(signals>0)&((close>=lower[params["long_from"]])& (close<bollinger_bands[:-1][:, -1]))] = 1.0  

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
