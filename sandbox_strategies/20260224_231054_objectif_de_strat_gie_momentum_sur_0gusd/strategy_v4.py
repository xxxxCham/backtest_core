from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'ema']

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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                # Define your logic for buy and sell signals here. Please replace with your desired conditions.
                long_mask = np.zeros(n, dtype=bool)  # initialize mask to track long positions
                short_mask = np.zeros(n, dtype=bool)  # initialize mask to track shorts

                # Define ATR calculation parameters if provided in params dictionary
                atr_param = params['atr'] or self.default_params["leverage"] * 0.2  
                atr = indicators['atr'][:n]  # get current bar's ATR for calculation

                # Define a buffer zone around price at which we will consider buying or selling depending on RSI value and ATR level.
                buy_price_buffer, sell_price_buffer = atr * 2., atr * 1.5  

                # Define conditions for buying signals here. Please replace with your desired logic.
                buy_conditions = [df['close'] > buy_price_buffer]   

                # Define conditions for selling signals here. Please replace with your desired logic.
                sell_conditions = [df['close'] < sell_price_buffer] 

                # Combine the above conditions to define a comprehensive signal mask.
                entry_mask = np.logical_or(long_mask, buy_conditions) & np.logical_or(short_mask, sell_conditions)   

                signals[entry_mask] = 1.0  

                # Apply ATR-based risk management rules here. Please replace with your desired conditions.
                if params['leverage'] > self.default_params["leverage"]:  # Apply leverage if it's greater than the default value
                    signals[entry_mask] *= (self.default_params["leverage"] / params['leverage'])  

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
