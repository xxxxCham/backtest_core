from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 1,
         'buffer': 0.2,
         'ema_period': 9,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=1,
                max_val=99,
                default=9,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=1,
                max_val=99,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=1,
                max_val=99,
                default=26,
                param_type='int',
                step=1,
            ),
            'buffer': ParameterSpec(
                name='buffer',
                min_val=0.0,
                max_val=0.5,
                default=0.2,
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
                long_mask = np.zeros(n, dtype=bool)
                short_mask = np.zeros(n, dtype=bool)

                # Implement explicit LONG / SHORT / FLAT logic here

                ema12 = indicators['ema'][-12:].values  # Use the last 12 EMA values for MACD computation
                ema26 = indicators['ema'].mean()[-2:].values  # Calculate the mean of the current and previous two EMA values to compute MACD

                indicators['macd']['macd'] = ema12 - ema26

                signal_line = indicators['macd'][-9:-3]

                short_signal = np.sum(np.abs((df[df['close']].diff() * 2)) < threshold) / n
                long_signal = (short_signal > self.params['leverage']) & signal_line[:-1][:n] <= indicators['macd']['macd'][:-1][:n]

                signals[(indicators['macd']['macd'] >= 0)][long_mask | short_mask] = 2 * ((df["close"].diff() - ema26) / (ema12 - ema26)) + 1
                signals[(indicators['macd']['macd'] < threshold)[short_mask]] = long_signal.values[short_mask].astype(int)

                # Write SL/TP columns into df if using ATR-based risk management
                sl_level = self.params['leverage'] * 1.5 * indicators['atr'][-9:-3][:n]
                tp_level = -sl_level[short_mask]

                signals[(indicators['macd']['macd'] < threshold)[long_mask]] = long_signal.values[~short_mask].astype(int)  # Long positions only when MACD crosses below signal line again after crossing above it
                signals[(indicators['macd']['macd'] >= 0)] *= (tp_level / sl_level)[long_mask]

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
