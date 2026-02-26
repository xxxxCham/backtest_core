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
        return ['atr']

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
        # Long position
        def long_position():
            macd, signal = generate_macd()  # Assuming `generate_macd` returns 'macd' and 'signal'

            rsi = get_rsi(df['close'])  # Assuming `get_rsi` returns a numpy array of 'rsi' values
            indicators['donchian']['upper'], indicators['donchian']['lower'] = generate_donchian()  # Assuming `generate_donchian` returns 'donchian_upper', 'np.nan_to_num(indicators['donchian']['middle'])', and 'donchian_lower'

            adx = generate_adx(df['high'], df['low'])  # Assuming `generate_adx` returns 'adx', 'plus_di', and 'minus_di'
            supertrend, direction = generate_supertrend()  # Assuming `generate_supertrend` returns 'supertrend', and 'direction' (not indicators['supertrend'] directly)

            stochastic = generate_stochastic(df.index, df['close'])  # Assuming `generate_stochastic` returns the indicators['stochastic']['stoch_k'] and indicators['stochastic']['stoch_d'] values

            indicators['bollinger']['upper'], indicators['bollinger']['middle'], indicators['bollinger']['lower'] = generate_bb()  # Assuming `generate_bb` returns 'bb_upper', 'bb_middle' and 'bb_lower'
        signals.iloc[:warmup] = 0.0
        return signals
