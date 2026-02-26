from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='obv_stochastic_atr_refined')

    @property
    def required_indicators(self) -> List[str]:
        return ['obv', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.0005,
         'leverage': 1,
         'stochastic_overbought': 80,
         'stochastic_oversold': 20,
         'stochastic_period': 14,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 2.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_period': ParameterSpec(
                name='stochastic_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_overbought': ParameterSpec(
                name='stochastic_overbought',
                min_val=70,
                max_val=95,
                default=80,
                param_type='int',
                step=1,
            ),
            'stochastic_oversold': ParameterSpec(
                name='stochastic_oversold',
                min_val=5,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.0001,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.9,
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
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Pull indicator arrays
        obv = np.nan_to_num(indicators['obv'])
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Previous OBV for trend check
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # Entry conditions
        long_mask = (
            (obv > prev_obv)
            & (indicators['stochastic']['stoch_k'] < params["stochastic_oversold"])
            & (atr > params["atr_threshold"])
        )
        short_mask = (
            (obv < prev_obv)
            & (indicators['stochastic']['stoch_k'] > params["stochastic_overbought"])
            & (atr > params["atr_threshold"])
        )

        # Exit condition (any factor reversal)
        exit_mask = (
            (obv < prev_obv)
            | (indicators['stochastic']['stoch_k'] > 50.0)
            | (atr < params["atr_threshold"])
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Ensure exits are flat
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
