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
        def generate_strategy(broker, datafeed, env):
            # Define parameters based on broker/data feed specifics

            ## Parameters to be defined by user:
            ## - indicators (list of strings): list of OHLCV indicator names
            ## - leverage (float): leverage ratio 
            ## - warmup (int): number of bars for warming up period
            ## - stop_loss_multiplier (float) and tp_multiplier (float): multiplier factors for SL/TP levels

            params = {
                'indicators': ['rsi', 'ema', 'atr'],  # Default set of indicators
                'leverage': 1,                      # Leverage ratio
                'warmup': 50,                       # Number of bars for warming up period
                'stop_loss_multiplier': 1.2,        # Multiplier factor for stop loss levels
                'tp_multiplier': 1.2                # Multiplier factor for take profit levels
            }

            ## Define custom logic to generate signals based on user inputs and parameters

            class BuilderGeneratedStrategy(StrategyBase):

                def __init__(self, broker, datafeed, env, params=None):
                    super().__init__(broker, datafeed, env)

                    if not params:  # Default parameters are defined in the function call above
                        params = self.params or {}

                    ## Load indicators from broker/data feed and apply OHLCV transformations (if any)

                def generate_signals(self):  
                    signals = pd.Series(0, index=df['close'].index, dtype=np.float64)  # Initialize empty signal array

                    ## Implement logic for entering long/short positions and exiting them based on generated signals

            return BuilderGeneratedStrategy(**params)
        signals.iloc[:warmup] = 0.0
        return signals
