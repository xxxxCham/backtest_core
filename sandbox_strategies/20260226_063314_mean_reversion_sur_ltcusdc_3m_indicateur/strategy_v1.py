from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ltc_stoch_ema_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_fast_period': 9,
         'ema_slow_period': 21,
         'leverage': 1,
         'stochastic_period': 14,
         'stochastic_smooth': 3,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast_period': ParameterSpec(
                name='ema_fast_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ema_slow_period': ParameterSpec(
                name='ema_slow_period',
                min_val=20,
                max_val=50,
                default=21,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        ema_fast = np.nan_to_num(indicators['ema'])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        atr = np.nan_to_num(indicators['atr'])

        # Long entry condition
        long_entry = (indicators['stochastic']['stoch_k'] < 20) & (ema_fast < np.roll(ema_fast, 1))
        long_mask = long_entry

        # Short entry condition
        short_entry = (indicators['stochastic']['stoch_k'] > 80) & (ema_fast > np.roll(ema_fast, 1))
        short_mask = short_entry

        # Long exit condition
        long_exit = (indicators['stochastic']['stoch_k'] > 80)

        # Short exit condition
        short_exit = (indicators['stochastic']['stoch_k'] < 20)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = df.loc[long_mask, "close"] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = df.loc[long_mask, "close"] + params["tp_atr_mult"] * atr[long_mask]

        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = df.loc[short_mask, "close"] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = df.loc[short_mask, "close"] - params["tp_atr_mult"] * atr[short_mask]

        if np.any(long_exit):
            signals[long_exit] = 0.0
        if np.any(short_exit):
            signals[short_exit] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
