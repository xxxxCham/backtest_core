from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_roc_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['momentum', 'roc', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'momentum_period': 12,
         'momentum_threshold': 10,
         'roc_period': 12,
         'roc_threshold': 10,
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
                default=12,
                param_type='int',
                step=1,
            ),
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'momentum_threshold': ParameterSpec(
                name='momentum_threshold',
                min_val=1,
                max_val=50,
                default=10,
                param_type='float',
                step=0.1,
            ),
            'roc_threshold': ParameterSpec(
                name='roc_threshold',
                min_val=1,
                max_val=50,
                default=10,
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
                min_val=1.0,
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

        # Wrap indicator arrays
        momentum = np.nan_to_num(indicators['momentum'])
        roc = np.nan_to_num(indicators['roc'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (momentum > params["momentum_threshold"]) & (
            roc > params["roc_threshold"]
        )
        short_mask = (momentum < -params["momentum_threshold"]) & (
            roc < -params["roc_threshold"]
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: cross down of momentum or roc below zero
        prev_momentum = np.roll(momentum, 1)
        prev_momentum[0] = np.nan
        prev_roc = np.roll(roc, 1)
        prev_roc[0] = np.nan
        exit_mask = ((momentum <= 0) & (prev_momentum > 0)) | (
            (roc <= 0) & (prev_roc > 0)
        )
        signals[exit_mask] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = (
                close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            )
            df.loc[long_mask, "bb_tp_long"] = (
                close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
            )
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = (
                close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            )
            df.loc[short_mask, "bb_tp_short"] = (
                close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
            )
        signals.iloc[:warmup] = 0.0
        return signals
