from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake_Case_Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=5,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
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
            n = len(df)  # length of the dataframe

            long_mask = np.zeros(n, dtype=bool)   # initialize signal for long positions
            short_mask = np.zeros(n, dtype=bool)  # initialize signal for short positions

            if self.default_params["leverage"] == 1:
                warmup = int(self.default_params['warmup'])  # replace with your actual parameter specs

                rsi = indicators[required_indicators[0]]   # replace with the name of the RSI indicator function you want to use
                bollinger = indicators[required_indicators[1]]  # also replace with the name of the Bollinger band indicator function you want to use

            entry_signal = np.zeros(n, dtype=bool)   # initialize signal for long positions
            short_mask = np.zeros(n, dtype=bool)   # initialize signal for short positions

            if len(df)>warmup:
                close = df["close"].values  # replace with your actual dataframe column name

                rsi_value = rsi['values'][0]  # get the first value of RSI data, you may need to adjust this depending on how your RSI function returns values

                if (rsi_value < self.default_params["warmup"]):   # replace with your actual condition for a new uptrend
                    long_mask[:warmup] = True  # set the first 'long' positions as true in the long_mask

            else:
                close = df["close"][0]  # use the very last price point as our initial entry signal for the market        

            if len(df)>warmup+1:  
                short_signal = (close<self.default_params['bollinger']['values'][int(rsi_value)])  # check Bollinger Bands conditions
                long_mask[int(rsi_value)]=short_signal[:-1]  # if condition is met, mark the corresponding entry as true in the mask

            return np.where((long_mask | short_mask)[:n], 1.0, 0.0)   # combine long and short signals with a 'buy' signal (1.0) or a 'sell' signal (0.0)
        signals.iloc[:warmup] = 0.0
        return signals
