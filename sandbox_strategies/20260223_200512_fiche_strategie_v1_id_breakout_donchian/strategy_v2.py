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
        return {'leverage': 1,
         'no_lookahead': True,
         'only_registry_indicators': True,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1,
                max_val=4,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2,
                max_val=8,
                default=6.0,
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
        def generate_signals(self, df, indicators, params):
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # implement explicit LONG / SHORT / FLAT logic
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            # Write SL/TP columns into df if using ATR-based risk management
            atr = np.nan_to_num(indicators['atr'])
            close = df['close'].values
            entry_mask = (signals == 1.0) & (close[:-1] > close[1:]) # Check for long entries
            signals[entry_mask | ((close[-1:] < close[:-1]) & (close - close[:(n-2)] > params['stop_atr']))] = -1.0  # Check for short entries

            stop_loss_mult = params.get('stop_atr', 1)
            tp_mult = params.get('tp_atr', 1)
            if signals[-1:] == -1.0:
                close_to_check = close[:-2] + (close[:(n-3)] / atr[(n-4):(n-2)]) # Check for TP level based on ATR and previous price levels

                entry_mask |= ((signals[-2:] == 1.0) & (close_to_check > close[:-(n-2)-1]))   # Check for long entries
            else:
                signals[-1] = 1.0     # Assume that the last signal is long if no stop loss reached yet

            entry_mask |= ((signals[:(n-3)] < -1.0) & (close[:-(n-2)-1] > close[(n-4):(n-1)])   # Check for short entries
                           | ((signals[-2:] == 1.0) & signals[-(n+1)::])                            # Check for long exit if TP reached and followed by new signal of -1.0
                          )
            signals[entry_mask] = (close[(n-3):(n)] / atr[:4]).apply(lambda x: abs(x)) > params['tp_percent'] * 2.5 # Check for target price based on ATR and recent price levels

            stop_loss_level = close[-1:] - signals[:(n-1)].values * stop_atr_mult * atr[(n-4):]
            if signals[:-(n+3)]:
                signals[signals.index >= (stop_loss_level + abs(close[-1:]))] = 0.0 # Check for exit long positions if price level reached with significant move in other direction

            close_to_check = np.concatenate((np.array([], dtype=float), signals[:-(n+3)]))
        signals.iloc[:warmup] = 0.0
        return signals
