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
        def generate_signals(df):
            indicators = {
                'rsi': df['close'].rolling(21).std(), # RSI calculation
                'ema': lambda x: pd.Series(x.ewm(com=3, min_periods=60).mean().iloc[:, 0], index=df.index), # EMA calculation
                'atr': lambda x: pd.Series(pd.cut(x['range'], bins=[2]*len(x)), index=df.index) # ATR calculation (using simple method without rolling window)
            }

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i in df['long_short'].unique():
                long_entries = [j[i] for j in signals if 'entry' in j and j['entry'][1]] # Long signals with entry bar 

                short_entries = [j[i] for j in signals if 'entry' in j and j['exit'][0]] # Short signals with exit bar 

                long_positions = (signals == 1) & (~isnan(long_entries)) # Long positions only at entry bars, ignore NaN values

                short_positions = ~(signals == -1) & (~np.isinf(short_entries)) # Short positions only at exit bars, ignore INF values

                signals[long_positions] += 1.0 * (i=='LONG') # Entree long si momentum haussier confirmé et risque contrôlé 

                signals[~long_positions & short_positions] -= 1.0 * (~i=='LONG' & i != 'INVALID') # Autrement, entree short si momentum baissier confirmé et risque contrôlé 

            for indicator in ['rsi', 'ema']:
                signals[indicator] = indicators[indicator]['close'] # Using pre-calculated EMA and RSI values from the dataframe

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
