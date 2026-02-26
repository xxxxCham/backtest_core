from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'macd', 'vortex']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'long_window': 19,
         'short_window': 9,
         'signal_line_period': 9,
         'stop_atr_mult': 1.5,
         'threshold': 4,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'short_window': ParameterSpec(
                name='short_window',
                min_val=7,
                max_val=21,
                default=9,
                param_type='int',
                step=1,
            ),
            'long_window': ParameterSpec(
                name='long_window',
                min_val=19,
                max_val=41,
                default=19,
                param_type='int',
                step=1,
            ),
            'signal_line_period': ParameterSpec(
                name='signal_line_period',
                min_val=7,
                max_val=21,
                default=9,
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
        def generate_signals(indicators, default_params):
            short_window = default_params['short_window']
            long_window = default_params['long_window']

            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for i in df.index:
                short_values = df.loc[i]['short'][short_window:]
                long_values = df.loc[i]['long'][:long_window]

                ema12, _, _ = indicators['ema'](short_values)
                macd, _, _ = indicators['macd'](np.concatenate([short_values, long_values]))
                vortex = indicators['vortex'].get('value', 0.0)

                # LONG intent: ema_cross(short_window, long_window), macd_cross_above_signal_line and vortex > threshold
                if np.abs(np.sum([macd[2], ema12[-1]]) - 0.) >= .05 or (len(long_values) > short_window // 2 and len(short_values) < long_window * 3 / 4):
                    signals[i] = np.float64(1.0)

                # SHORT intent: ema_cross(short_window, long_window), macd_cross_below_signal_line and vortex < -threshold
                elif (np.abs(np.sum([macd[2], ema12[-1]]) + 3 * np.std(long_values)) / len(long_values) <= .05 or \
                      (len(short_values) > short_window // 2 and len(long_values) < long_window * 3 / 4)):
                    signals[i] = -np.float64(1.0)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
