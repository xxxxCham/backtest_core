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
        # You need to define function generate_signals which takes dataframe as input and returns a Series of signals where each element is either 1 (long), -1 (short) or 0.
        # Also, you need to specify the list `indicators` available in this method: ['rsi', 'ema', 'atr']

        def generate_signals(df):
            # Define default parameters for signals
            default_params = {'leverage': 1}

            # Iterate over each row (bar) of the dataframe
            for i, bar in df.iterrows():
                # Get indicator values using correct syntax and store them as numpy arrays

                rsi_val = ...
                ema_val = ...
                atr_val = ...

                # Check if momentum is bullish (long) or bearish (short), considering ATR-based stop loss and take profit levels.

                # Use correct syntax to access indicators arrays by using arr[i] or vectorized masks
                rsi_arr = rsi_val[...].tolist()  # Assuming the input is a numpy array
                ema_arr = ...   # Assuming the input is a numpy array

                atr_arr = atr_val[...].tolist()    # Assuming the input is a numpy array

                # Check if momentum (using RSI, EMA or ATR) confirms and risk level controlled. 

                # Use correct syntax to access indicators arrays by using arr[i] or vectorized masks

                rsi_confirm = ...   # Assuming the input is a numpy array
                ema_confirm = ...    # Assuming the input is a numpy array
                atr_confirm = ...    # Assuming the input is a numpy array

                # Use correct syntax to access indicators arrays by using arr[i] or vectorized masks

                rsi_risk = ...   # Assuming the input is a numpy array
                ema_risk = ...    # Assuming the input is a numpy array
                atr_risk = ...     # Assuming the input is a numpy array

                if (rsi_confirm > rsi_arr[...]) and (ema_confirm > ema_arr[...]):  
                    signals[...] = 1.0    # Long signal when momentum confirms, risk controlled
                elif (rsi_risk < rsi_arr[...]) or (atr_risk <= atr_arr[...]):
                    signals[...] = -1.0   # Short signal when momentum does not confirm, risk uncontrolled
                else: 
                    signals[...] = 0.0    # No trade signal when neither condition is met


            return signals
        signals.iloc[:warmup] = 0.0
        return signals
