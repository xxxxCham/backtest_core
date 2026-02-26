from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

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
        def generate_signals(df, default_params):
            # Calculate Bollinger Bands Upper and Lower bands. 
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
                bollinger(df['close'].values, df['period'], window=default_params[0])

            # Get RSI values for each security (using numpy arrays)
            rsii = rsi(df['close'].values, timeperiod=5).values.tolist()

            # Calculate ATR based on close prices and default_params[1] is the number of lookback periods to use in ATR calculations
            indicators['atr'] = atr(df['close'], window_size=default_params[0]).values.tolist()[0][-1]

            # Calculate Donchian Channels Upper, Lower bands and Donchian Bands Middle using default params [2][3]. 
            indicators['donchian']['upper'] = np.roll(donchian_bands['upper'].values, -default_params[2])
            indicators['donchian']['middle'] = np.roll(donchian_bands['mid'].values, -default_params[2])

            # Calculate ADX values for each security (using numpy arrays) 
            adx = adx(df['high'], df['low'], df['close']).values.tolist()[-1]

            # Use the Supertrend and Stochastic indicators to generate signals based on their default parameters
        signals.iloc[:warmup] = 0.0
        return signals
