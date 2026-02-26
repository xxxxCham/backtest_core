from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr']

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
        def generate_signals(indicators):
            df = pd.DataFrame() # replace this line with actual data frame

            for i, row in df.iterrows():
                close = row['close']

                indicators['bollinger']['upper'] = np.roll(indicators['donchian']['upper'][i], 1)
                donchian_breakdown = (close > indicators['bollinger']['upper']).astype(int) & (indicators['bollinger']['lower'][i] < close).astype(int)

                adx, indicators['adx']['plus_di'], indicators['adx']['minus_di'] = indicators['adx']['adx'][i], indicators['adx']['plus_di'][i], indicators['adx']['minus_di'][i]
                adx_and_others = (adx[i] > 25) & ((indicators['adx']['plus_di'][i] > indicators['adx']['minus_di'][i]) | np.isnan(indicators['adx']['plus_di'][i])) # for 'AND' condition, use AND operator with both sides as boolean masks

                indicators['stochastic']['stoch_k'] = indicators['stochastic']['stoch_k'][i] - (indicators['stochastic']['stoch_k'][i] - indicators['stochastic']['stoch_d'][i]) * 9/10 # for 'AND' condition, use AND operator with both sides as boolean masks

                if donchian_breakdown.any() and adx_and_others: # for 'LONG' intent
                    signals[i] = 1.0

                elif not donchian_breakdown.any() or (donchian_breakdown & ~adx_and_others): # for 'SHORT' intent
                    signals[i] = -1.0

                else: # no conditions met, keep signal 0
                    signals[i] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
