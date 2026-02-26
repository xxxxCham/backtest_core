from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGY v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'atr']

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
        def generate_signals(indicators, default_params={}, bars=None):
            df = bars if bars is not None else pd.DataFrame()  # Assuming 'bars' is a DataFrame of OHLC data

            # Define default parameters for EMA and ATR
            ema_fast_length = 21
            ema_slow_length = 50
            atr_multiplier = 1.5

            signals = pd.Series(np.nan, index=df.index, dtype=np.float64)

            for i in df.index:
                # SuperTrend direction
                superTrendDirection = indicators['supertrend'][i]

                if np.isnan(superTrendDirection):  # Skip bars without supertrend data
                    continue

                adxDirection = indicators['adx']['adx'] + indicators['adx']['plus_di']/100 + indicators['adx']['minus_di']/100

                if np.isnan(adxDirection) or superTrendDirection == 1 and (SHORT <= adxDirection <= LONG):  # Long signal
                    signals[i] = 1.0

                elif np.isnan(adxDirection) or superTrendDirection == -1 and (LONG < adxDirection <= SHORT):  # Short signal
                    signals[i] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
