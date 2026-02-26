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
        return ['bollinger', 'donchian', 'atr']

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
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=4.0,
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
        def generate_signals(df, indicators, params):
            # Convert dict to Series (required by pd.Series)
            df = pd.Series(data=dict2list(indicators), index=df.index)

            close = df['close']
            high = df['high']
            low = df['low']

            upper_bb, middle_bb, lower_bb = indicators[0], indicators[1], indicators[2]
            n_stddevs = len(upper_bb) - 1

            # Calculate Bollinger Bands and Donchian Channels
            bollinger_band = pd.Series(pd.stats.binnings.bin_edge(close, np.arange(0.05, 0.95, 0.05)), index=df.index)

            # Calculate ADX for each bar and the final result
            upper_band = bollinger_band['upper']
            middle_band = bollinger_band['middle']
            lower_band = bollinger_band['lower']

            adx, _, _ = indicators[2]
        signals.iloc[:warmup] = 0.0
        return signals
