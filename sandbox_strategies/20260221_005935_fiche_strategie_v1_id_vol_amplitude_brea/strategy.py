from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEGIE v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['amplitude_hunter', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.5,
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
        def generate_signals(df):
            # Create empty dataframe for signals
            signals = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float64)

            # Iterate over each row in the DataFrame
            for i, row in df.iterrows():
                close_val = row['close']

                # For each available indicator...
                for name, value in indicators.items():
                    if 'score' not in value:
                        raise ValueError(f"Indicator {name} missing score.")

                    # Check the conditions and update signal
                    if (value[0]['amplitude_hunter'].score > 0.6) & (row['close'] > row['donchian']) or \
                       (value[0]['amplitude_hunter'].score > 0.6) & (row['close'] < row['donchian']):
                        signals[i] = 1.0    # Buy signal
                    elif (value[0]['amplitude_hunter'].score <= 0.6) and \
                          (row['close'] >= value[0]['donchian'].lower or row['close'] < value[0]['donchian'].upper):
                        signals[i] = -1.0   # Sell short signal
                    else:
                        signals[i] = 0.0    # Do nothing signal

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
