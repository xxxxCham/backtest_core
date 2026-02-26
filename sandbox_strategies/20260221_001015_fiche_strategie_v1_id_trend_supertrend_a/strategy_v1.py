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
        return ['supertrend', 'adx', 'atr']

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
                min_val=0.1,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0,
                max_val=10.0,
                default=5.5,
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
            # Define default parameters for ATR-based SL/TP
            df['bb_stop_long'], df['bb_tp_long'] = bb_atrs(df)['donchian'][0], \
                                                  bb_atrs(df)['donchian'][1]
            df['bb_stop_short'], df['bb_tp_short'] = bb_atrs(df)['donchian'][2], \
                                                     bb_atrs(df)['donchian'][3]

            # Get indicators data
            supertrend, adx, atr = [get_indicators(i) for i in ['supertrend', 'adx', 'atr']]

            # Check the conditions to generate signals
            long_intent = (indicators['supertrend'].values == 1.0 and \
                           np.abs(indicators['adx'].values) > 25.0 and \
                           indicators['supertrend'].values[-1] == -1.0 and \
                           np.abs(indicators['adx'].values) > 25.0)

            short_intent = (indicators['supertrend'].values == 1.0 and \
                            np.abs(indicators['adx'].values) < 20.0 and \
                            indicators['supertrend'].values[-1] == -1.0 and \
                            np.abs(indicators['adx'].values) < 20.0)

            signals = pd.Series([0]*len(df), index=df.index, dtype=np.float64)

            # Generate signals based on the conditions
            signals[long_intent] = 1.0
            signals[short_intent] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
