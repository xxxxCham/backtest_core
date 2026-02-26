from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_stoch_atr_scalp')

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
         'stop_atr_mult': 0.8,
         'tp_atr_mult': 2.0,
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
                min_val=5,
                max_val=20,
                default=5,
                param_type='int',
                step=1,
            ),
            'stoch_d_period': ParameterSpec(
                name='stoch_d_period',
                min_val=3,
                max_val=10,
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
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=0.8,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
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
        # initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # extract indicator arrays
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(indicators['stochastic']["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])

        # previous values for slope and cross detection
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        prev_k = np.roll(indicators['stochastic']['stoch_k'], 1)
        prev_k[0] = np.nan
        prev_d = np.roll(indicators['stochastic']['stoch_d'], 1)
        prev_d[0] = np.nan

        # cross functions
        cross_up_kd = (indicators['stochastic']['stoch_k'] > indicators['stochastic']['stoch_d']) & (prev_k <= prev_d)
        cross_down_kd = (indicators['stochastic']['stoch_k'] < indicators['stochastic']['stoch_d']) & (prev_k >= prev_d)

        # long entry conditions
        long_mask = (
            (close > ema)
            & (ema > prev_ema)
            & (indicators['stochastic']['stoch_k'] < 20)
            & cross_up_kd
        )

        # short entry conditions
        short_mask = (
            (close < ema)
            & (ema < prev_ema)
            & (indicators['stochastic']['stoch_k'] > 80)
            & cross_down_kd
        )

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # enforce warmup period (rule 10)
        signals.iloc[:50] = 0.0

        # write ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # long entries
        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - params["stop_atr_mult"] * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + params["tp_atr_mult"] * atr[long_entries]

        # short entries
        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + params["stop_atr_mult"] * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - params["tp_atr_mult"] * atr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals
