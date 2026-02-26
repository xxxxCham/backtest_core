from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_roc_atr_5m')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'roc', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'momentum_period': 10,
         'momentum_threshold': 0.0,
         'roc_period': 10,
         'stop_atr_mult': 2.1,
         'tp_atr_mult': 5.3,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'momentum_threshold': ParameterSpec(
                name='momentum_threshold',
                min_val=0,
                max_val=1,
                default=0.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=5.3,
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

        signals.iloc[:warmup] = 0.0

        momentum = np.nan_to_num(indicators['momentum'])
        roc = np.nan_to_num(indicators['roc'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        long_mask = (momentum > params["momentum_threshold"]) & (roc > 0.0)
        short_mask = (momentum < -params["momentum_threshold"]) & (roc < 0.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
