from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_v1')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 35,
         'rsi_period': 13,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 5.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=13,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.0,
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

        # wrap indicator arrays
        close = df["close"].values
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        bb = indicators['bollinger']
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])

        # helper for cross_any
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # entry conditions
        long_mask = (
            (close < indicators['bollinger']['lower'])
            & (rsi < params["rsi_oversold"])
            & (adx_val < 25.0)
        )
        short_mask = (
            (close > indicators['bollinger']['upper'])
            & (rsi > params["rsi_overbought"])
            & (adx_val < 25.0)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit conditions (not used directly in signals but can be used for position management)
        exit_mask = (
            cross_any(close, indicators['bollinger']['middle'])
            | cross_any(rsi, np.full(n, 50.0))
            | (adx_val > 25.0)
        )
        # ensure exits don't overwrite entries
        signals[exit_mask & ~long_mask & ~short_mask] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = (
                close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            )
            df.loc[long_mask, "bb_tp_long"] = (
                close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
            )

        # ATR-based SL/TP for short entries
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = (
                close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            )
            df.loc[short_mask, "bb_tp_short"] = (
                close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
            )
        signals.iloc[:warmup] = 0.0
        return signals
