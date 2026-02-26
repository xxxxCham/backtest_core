from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Vortex_ICHIMOKU_SMA')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'ichimoku', 'sma']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'risk_ratio': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'warmup': ParameterSpec(
                name='warmup',
                min_val=0,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'risk_ratio': ParameterSpec(
                name='risk_ratio',
                min_val=2.0,
                max_val=4.0,
                default=3.0,
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
        def generate_signals(self, df, indicators, params):
            # Initialize signals to all zeros with dtype float64
            signals = pd.Series(0., index=df.index, dtype=np.float64)

            # Loop over each bar in the dataframe
            for i, row in df.iterrows():
                close_price = row['close']

                # Compute ATR based on 14 periods
                atr = indicators['vortex'].atk_atr(params=params)

                if len(signals) < params['warmup']:
                    signals[i] = 0.

                elif i >= params['warmup'] and i <= params['warmup'] + params['training_days']:
                    # Long position logic here
                    signals[i] = row['close'] - atr * params['multiplier_atr']

                else:
                    # Short position logic here
                    signals[i] = row['close'] + atr * params['multiplier_atr']

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
