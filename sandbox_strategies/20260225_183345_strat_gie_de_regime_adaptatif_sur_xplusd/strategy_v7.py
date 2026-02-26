from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='regime_adaptive_xplusdc_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'atr_threshold': 1.0,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
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

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # extract indicator arrays
        close = df["close"].values
        open_ = df["open"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_arr = np.nan_to_num(indicators['adx']["adx"])
        obv = np.nan_to_num(indicators['obv'])
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan

        # entry conditions
        long_mask = (
            (close > open_)
            & (adx_arr > 25)
            & (obv > obv_prev)
            & (atr > params["atr_threshold"])
        )
        short_mask = (
            (close < open_)
            & (adx_arr > 25)
            & (obv < obv_prev)
            & (atr > params["atr_threshold"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions
        exit_mask = (atr < params["atr_threshold"]) | (adx_arr < 20)
        signals[exit_mask] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
