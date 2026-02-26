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
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'capital': 10000.0,
         'fees': 10.0,
         'leverage': 1,
         'slippage': 5.0,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=3.0,
                max_val=7.0,
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
        def generate_signals(df, default_params={}, indicators=None):
            # Check if indicators are provided. If not, use the available ones
            if indicators is None:
                indicators = {'atr': df['close'].rolling(window=20).std(), 'donchian': df['close'].rolling(window=5).mean()}

            # Calculate Donchian Bands (Upper and Lower)
            indicators['donchian']['upper'], indicators['donchian']['lower'] = indicators['donchian']['middle'], indicators['donchian']['lower']

            # Bollinger Band Strategy: Close > Donchian Middle AND indicators['adx']['adx'] > 35
            signals['bollinger'][(df['close'] > indicators['donchian']['upper']) & (indicators['adx'].ADX > 35)] = 1.0

            # ATD strategy: Close < Donchian Lower AND indicators['adx']['adx'] > 35
            signals['atd'][(df['close'] < indicators['donchian']['lower']) & (indicators['adx'].ADX > 35)] = -1.0

            # RSI strategy: not implemented yet, we will use the code from another class later
            # signals['rsi'] = rsi(df)
        signals.iloc[:warmup] = 0.0
        return signals
