from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vol_amplitude_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['amplitude_hunter']

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
            'donchian_channel.period': ParameterSpec(
                name='donchian_channel.period',
                min_val=20,
                max_val=100,
                default=40,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.5,
                max_val=3,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'target_atr_multi': ParameterSpec(
                name='target_atr_multi',
                min_val=2,
                max_val=5,
                default=3,
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
        def generate_signals(indicators):
            df = pd.DataFrame() # Assume we have a DataFrame object here

            # LONG intent
            mask1 = indicators['amplitude_hunter'].score > 0.9 and df["close"] > donchian[3]
            signal1 = np.where((mask1, True), 1.0, 0.0)
            signals['long'] += signal1

            # SHORT intent
            mask2 = indicators['amplitude_hunter'].score < 0.15 and df["close"] < donchian[3]
            signal2 = np.where((mask2, True), -1.0, 0.0)
            signals['short'] += signal2

            # ATR-based SL/TP entries
            atr_series = indicators['atr'].get('average', df)
            upper_band = donchian[3] + (atr_series * 1).shift()  
            lower_band = donchian[0] - (atr_series * 1).shift()    

            mask3 = ((df["close"] > upper_band) | indicators['amplitude_hunter'].score > 0.9 ) & \
                    (df["close"] < donchian[-2])
            signal3 = np.where((mask3, True), 1.0, 0.0)
            signals['long'] += signal3

            mask4 = ((df["close"] > lower_band ) | indicators['amplitude_hunter'].score < 0.15) & \
                    (df["close"] < donchian[-2])
            signal4 = np.where((mask4, True), -1.0, 0.0)
            signals['short'] += signal4
        signals.iloc[:warmup] = 0.0
        return signals
