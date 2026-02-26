from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx.multiplier': '2.5',
         'leverage': 1,
         'rsi.overbought': '80',
         'rsi.oversold': '30',
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

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

            # implement explicit LONG / SHORT / FLAT logic
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            sl_tp_levels = df['close'] - params["stop_atr"] * indicators['atr'][:, np.newaxis]\
                + df['close'] + params["take_profit"] * indicators['atr'][:, np.newaxis] # compute SL/TP levels based on atr, close and take profit parameters
            entry_mask = (df['close'] > sl_tp_levels[0]) & \
                         (df['close'].shift() <= sl_tp_levels[1])\
                         &(df['close'].shift(-1) >= sl_tp_levels[-2]) # compute long/short signals based on SL/TP levels and previous close price
            long_mask = entry_mask & ~signals.iloc[:warmup] # combine all filters to get final signal

            signals[long_mask] = 1.0   # set signals for LONG positions
            signals[~long_mask] = -1.0  # set signals for SHORT positions or no position at all

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
