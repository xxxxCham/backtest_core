from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='xplusdc_30m_regime_adaptive_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'atr_threshold': 0.2,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.05,
                max_val=1.0,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.4,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=50,
                default=30,
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
        # Prepare indicator arrays
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        obv = np.nan_to_num(indicators['obv'])
        close = df["close"].values
        open_ = df["open"].values

        # Boolean masks for entries
        long_mask = (
            (atr > params["atr_threshold"])
            & (adx_val > 25)
            & (obv > 0)
            & (close > open_)
        )
        short_mask = (
            (atr > params["atr_threshold"])
            & (adx_val > 25)
            & (obv < 0)
            & (close < open_)
        )
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_close = np.roll(close, 1)
        prev_open = np.roll(open_, 1)
        prev_close[0] = np.nan
        prev_open[0] = np.nan
        cross_any = (
            ((close > open_) & (prev_close <= prev_open))
            | ((close < open_) & (prev_close >= prev_open))
        )
        exit_mask = ((atr > params["atr_threshold"]) & (adx_val < 20)) | cross_any
        signals[exit_mask] = 0.0

        # Warm‑up protection
        signals.iloc[:50] = 0.0

        # ATR‑based stop‑loss and take‑profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
