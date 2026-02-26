from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        # Indicators available in this method: ['macd', 'rsi', 'atr']
        def generate_signals(indicators, macd, signal):
            # LONG intent: cross_up(mac.macd, mac.signal) AND rsi > 45 AND rsi < 65

            long = (indicators['macd'].macd + indicators['macd'].signal) > indicators['macd']['signal'] \
                    and indicators['rsi'] > 45 \
                    and indicators['rsi'] < 65

            # LONG_LONG intent: cross_up(mac.macd, mac.signal), rsi > 45 AND rsi < 65

            long_long = (indicators['macd'].macd + indicators['macd'].signal) > indicators['macd']['signal'] \
                    and indicators['rsi'] > 45 \
                    and indicators['rsi'] < 65

            # LONG_SHORT intent: cross_down(mac.macd, mac.signal), rsi <= 30 AND rsi >= 60

            long_short = (indicators['macd'].macd + indicators['macd'].signal) > indicators['macd']['signal'] \
                    and indicators['rsi'] <= 30 \
                    and indicators['rsi'] >= 60

            # SHORT intent: cross_up(mac.macd, mac.signal), rsi > 50 AND rsi < 70

            short = (indicators['macd'].macd + indicators['macd'].signal) > indicators['macd']['signal'] \
                    and indicators['rsi'] > 50 \
                    and indicators['rsi'] < 70

            # SHORT_SHORT intent: cross_down(mac.macd, mac.signal), rsi <= 60 AND rsi >= 40

            short_short = (indicators['macd'].macd + indicators['macd'].signal) > indicators['macd']['signal'] \
                    and indicators['rsi'] <= 60 \
                    and indicators['rsi'] >= 40

            # Assign signals based on conditions:
            signals = np.select([long, long_short, long_long], [1.0, -1.0, 0.0])
            signals = np.select([short, short_short], [-1.0, 0.0])

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
