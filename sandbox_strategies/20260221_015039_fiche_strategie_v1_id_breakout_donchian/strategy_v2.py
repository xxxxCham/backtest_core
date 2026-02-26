from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.25, 'tp_atr_mult': 3.5, 'warmup': 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.75,
                max_val=4.5,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.3,
                max_val=6.5,
                default=3.5,
                param_type='float',
                step=0.1,
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
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Initialize boolean masks for long and short positions
            long_mask = np.zeros(len(signals), dtype=bool)
            short_mask = np.zeros(len(signals), dtype=bool)

            n = len(df)

            stop_sl_level = params['leverage'] * 1.05   # Set Stop Loss level at 1.05 times leverage
            stop_tp_level = params['leverage'] / 2.0   # Set Take Profit level at half of leverage

            atr = indicators['atr']                    # Get ATR indicator

            entry_mask = (df > df.rolling(window=self.k_donchian).mean() - self.k_donchian * atr) & \
                         (df < df.rolling(window=self.k_donchian).mean() + self.k_donchian * atr)  # Donchian channel breakout logic

            long_mask[entry_mask] = True                    # Set long positions to True in the entry mask
            short_mask[~entry_mask] = False                 # Set short positions to False in non-breakout mask

            signals[(signals == 1) | (long_mask)] = -params['leverage']   # Set long entries to -Leverage level
            signals[(~(signals > 0)) & (~short_mask)] = params['leverage']  # Set short positions to +Leverage level when not in a long position

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
