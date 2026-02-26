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
        
        # Extract indicators
        rsi_values = indicators['rsi']
        ema_values = indicators['ema']  # Assuming ema contains multiple EMAs
        atr_values = indicators['atr']
        
        # Calculate rolling mean
        rolling_mean = df['close'].rolling(window=21).mean().values
        
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        
        # === LONG ENTRY LOGIC ===
        confirmed_bullish_momentum = (rsi_values < 70) & \
                                    (ema_values[21] < ema_values[50]) & \
                                    (df['close'].values > rolling_mean)
        controlled_risk = atr_values > 2.0
        long_entry_conditions = confirmed_bullish_momentum & controlled_risk
        long_mask[long_entry_conditions] = True
        
        # === SHORT ENTRY LOGIC ===
        confirmed_bearish_momentum = (rsi_values > 30) & \
                                     (ema_values[21] > ema_values[50]) & \
                                     (df['close'].values < rolling_mean)
        short_entry_conditions = confirmed_bearish_momentum & controlled_risk
        short_mask[short_entry_conditions] = True
        
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        
        return signals