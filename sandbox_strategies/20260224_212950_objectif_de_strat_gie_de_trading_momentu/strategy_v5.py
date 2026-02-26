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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, any], params:Dict[str,any]) ->pd.Series:
            signals = pd.Series(0.0, index=df.index)
            n = len(df)

            # Implement ATR based stop loss and take profit logic here (if applicable). 
            # For now we assume that the strategy is long only with no risk management.
            atr_periods = params['atr'].value if 'atr' in params else 14  
            stop_loss_mult = params['stop_loss_multiplier'].value if 'stop_loss_multiplier' in params else 0.5 # Assume a 50% trailing stop loss is used

            long_mask = df['close'].pct_change() < -atr_periods*params['atr'].value * stop_loss_mult  
            short_mask = (df['close'] > params["rsi"].value) & (df.shift(-1)['close'] >= params["rsi"].value) # Assume RSI has signaled a long position

            signals[long_mask] = 1.0
            signals[short_mask] = -1.0
            signals[~long_mask & ~short_mask] = 0.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
