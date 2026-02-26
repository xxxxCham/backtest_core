from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_stoch_atr_scalp_improved')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stoch_d_period': 3,
         'stoch_k_period': 5,
         'stoch_smooth_k': 3,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.2,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stoch_k_period': ParameterSpec(
                name='stoch_k_period',
                min_val=3,
                max_val=10,
                default=5,
                param_type='int',
                step=1,
            ),
            'stoch_d_period': ParameterSpec(
                name='stoch_d_period',
                min_val=2,
                max_val=8,
                default=3,
                param_type='int',
                step=1,
            ),
            'stoch_smooth_k': ParameterSpec(
                name='stoch_smooth_k',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=60,
                default=30,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Indicator arrays
        ema = indicators['ema']
        indicators['stochastic']['stoch_k'] = indicators['stochastic']['stoch_k']
        indicators['stochastic']['stoch_d'] = indicators['stochastic']['stoch_d']
        atr = indicators['atr']
        close = df['close'].values

        # Mean ATR for comparison
        atr_mean = np.mean(atr)

        # Cross detection (avoid wrap-around at start)
        cross_up = (indicators['stochastic']['stoch_k'] > indicators['stochastic']['stoch_d']) & (np.roll(indicators['stochastic']['stoch_k'], 1) <= np.roll(indicators['stochastic']['stoch_d'], 1))
        cross_down = (indicators['stochastic']['stoch_k'] < indicators['stochastic']['stoch_d']) & (np.roll(indicators['stochastic']['stoch_k'], 1) >= np.roll(indicators['stochastic']['stoch_d'], 1))
        cross_up[0] = False
        cross_down[0] = False

        # Long and short signal masks
        long_mask = (close > ema) & (indicators['stochastic']['stoch_k'] < 20) & cross_up & (atr > atr_mean)
        short_mask = (close < ema) & (indicators['stochastic']['stoch_k'] > 80) & cross_down & (atr > atr_mean)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:warmup] = 0.0
        return signals
