from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adx_stoch_atr_trend_entry')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stochastic_smooth': 3,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.8,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=2,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_smooth': ParameterSpec(
                name='stochastic_smooth',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=2.8,
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

        # Extract indicator arrays
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(indicators['stochastic']["stoch_d"])
        atr_arr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (adx_arr > 25) & (indicators['stochastic']['stoch_k'] > indicators['stochastic']['stoch_d'])
        short_mask = (adx_arr > 25) & (indicators['stochastic']['stoch_k'] < indicators['stochastic']['stoch_d'])

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_mult = params.get("stop_atr_mult", 1.2)
        tp_mult = params.get("tp_atr_mult", 2.8)

        # Long entries
        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_mult * atr_arr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_mult * atr_arr[long_entries]

        # Short entries
        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_mult * atr_arr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_mult * atr_arr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals
