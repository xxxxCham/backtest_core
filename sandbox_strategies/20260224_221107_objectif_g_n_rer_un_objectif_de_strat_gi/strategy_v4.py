from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ALTUSDC_Strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'donchian_middle_period': 14,
         'leverage': 2,
         'macd_fast_period': 14,
         'macd_signal_period': 9,
         'macd_slow_period': 3,
         'signal_stochastic_period': 9,
         'slow_stochastic_period': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_middle_period': ParameterSpec(
                name='donchian_middle_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
            ),
            'slow_stochastic_period': ParameterSpec(
                name='slow_stochastic_period',
                min_val=3,
                max_val=15,
                default=3,
                param_type='int',
                step=1,
            ),
            'signal_stochastic_period': ParameterSpec(
                name='signal_stochastic_period',
                min_val=9,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=8,
                max_val=15,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=3,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        def generate_signals(df, default_params):
            # Define Donchian bands and Bollinger Bands.
            donchian = df['close'].rolling(window=20).max()
            indicators['bollinger']['upper'] = 0.01 * (df['close'] - np.mean(donchian)) + \
                np.mean(donchian)
            indicators['bollinger']['middle'] = 0.01 * (np.mean(donchian) - donchian[0]) + \
                donchian[0]
            indicators['bollinger']['lower'] = 0.01 * (-donchian[-1] + np.mean(donchian)) + \
                np.mean(donchian)

            # Calculate the Stochastic K and D values, and the MACD signal.
            stochastic_k = (df['close'] - df['low']) / (df['high'] - df['low'])
            stochastic_d = 0.01 * (stochastic_k[len(stochastic_k) - 1] - \
                stochastic_k[:-2].mean()) + stochastic_k[:-2].mean()
            macd = ta.MACD.from_abar(df, fastperiod=12, slowperiod=26, signalperiod=9)

            # Calculate the Donchian Middle and previous upper band of Bollinger Bands.
            indicators['donchian']['middle'] = 0.01 * (donchian[-1] - donchian[0]) + \
                donchian[0] if len(df['close']) > 2 else np.nan

            # Calculate the relative strength index and true range.
            rsi_values = ta.RSI(df, timeperiod=14)
            tr_values = ta.ATR(df).diff() * np.sqrt(len(tr_values)) - \
                ta.ATR(df).diff()[-1]

            # Calculate the true range and average true range.
        signals.iloc[:warmup] = 0.0
        return signals
