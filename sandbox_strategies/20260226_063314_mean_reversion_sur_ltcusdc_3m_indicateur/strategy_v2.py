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
        return {'ema_long_period': 21,
         'ema_short_period': 9,
         'leverage': 1,
         'stoch_d_period': 3,
         'stoch_k_period': 3,
         'stoch_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_short_period': ParameterSpec(
                name='ema_short_period',
                min_val=5,
                max_val=30,
                default=9,
                param_type='int',
                step=1,
            ),
            'ema_long_period': ParameterSpec(
                name='ema_long_period',
                min_val=20,
                max_val=50,
                default=21,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
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

        ema_long = np.nan_to_num(indicators['ema'])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])

        # Long condition
        long_condition = (indicators['stochastic']['stoch_k'] > 20) & (ema_long > ema_long)  # Always True to prevent errors
        long_mask = (indicators['stochastic']['stoch_k'] > 20) & (np.roll(ema_long, 1) < ema_long)

        # Short condition
        short_condition = (indicators['stochastic']['stoch_k'] < 80) & (ema_long < ema_long) # Always True to prevent errors
        short_mask = (indicators['stochastic']['stoch_k'] < 80) & (np.roll(ema_long, 1) > ema_long)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP (example)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        close = df["close"].values
        entry_mask = (signals == 1.0) | (signals == -1.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan

        if np.any(entry_mask):
            df.loc[entry_mask & (signals == 1.0), "bb_stop_long"] = close[entry_mask & (signals == 1.0)] - stop_atr_mult * atr[entry_mask & (signals == 1.0)]
            df.loc[entry_mask & (signals == 1.0), "bb_tp_long"] = close[entry_mask & (signals == 1.0)] + tp_atr_mult * atr[entry_mask & (signals == 1.0)]
        signals.iloc[:warmup] = 0.0
        return signals
