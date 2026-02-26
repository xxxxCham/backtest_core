from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

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
        # define the condition for LONG and SHORT trades
        long_trade_condition = {
            'macd': {'cross_up': False, 'signal': False}, 
            'rsi': {'above_60': True, 'below_40': True}, 
            'atr': {'above_10': True}
        }
        short_trade_condition = {
            'macd': {'cross_down': False, 'signal': False}, 
            'rsi': {'above_60': True, 'below_40': True}, 
            'atr': {'above_10': True}
        }

        # generate the body lines for each trade condition
        def generate_long_trade():
            # calculate macd and signal values

            # check if conditions are met (macd above zero, rsi in overbought region)
            return long_trade_condition['macd']['cross_up'] & \
                   long_trade_condition['rsi']['above_60'] & \
                   long_trade_condition['atr']['above_10'] 

        def generate_short_trade():
            # calculate macd and signal values

            # check if conditions are met (macd below zero, rsi in oversold region)
            return short_trade_condition['macd']['cross_down'] & \
                   short_trade_condition['rsi']['below_40'] & \
                   short_trade_condition['atr']['above_10'] 

        # generate the body lines for each signal assignment (0.0 or -1.0)            
        def assign_signals():  
            signals = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float64)

            if np.any((generate_long_trade(), generate_short_trade())):  # check for long trades with conditions met
                signals[conditions['LONG']] = -1.0  

            elif np.any(generate_long_trade() & ~generate_short_trade()):  # check for short trades with conditions met
                signals[conditions['SHORT']] = -1.0  

        # return the generated signals series as a dictionary
        def generate():
            assign_signals()
            return {name: value for name, value in signals.items()}
        signals.iloc[:warmup] = 0.0
        return signals
