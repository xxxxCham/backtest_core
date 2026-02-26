from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_trend_filtered')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 20,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 3.0,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
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

        # helper for cross_any
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y) | (x < y) & (prev_x >= prev_y)

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_d = indicators['adx']
        adx = np.nan_to_num(adx_d["adx"])
        atr = np.nan_to_num(indicators['atr'])

        long_mask = (close > upper) & (rsi > params["rsi_overbought"]) & (adx > 25)
        short_mask = (close < lower) & (rsi < params["rsi_oversold"]) & (adx > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        cross_close_mid = cross_any(close, middle)
        cross_rsi_50 = cross_any(rsi, np.full_like(rsi, 50.0))
        adx_exit = adx < 20
        exit_mask = cross_close_mid | cross_rsi_50 | adx_exit
        signals[exit_mask] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entry_mask = signals == 1.0
        short_entry_mask = signals == -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = (
            close[long_entry_mask] - params["stop_atr_mult"] * atr[long_entry_mask]
        )
        df.loc[long_entry_mask, "bb_tp_long"] = (
            close[long_entry_mask] + params["tp_atr_mult"] * atr[long_entry_mask]
        )
        df.loc[short_entry_mask, "bb_stop_short"] = (
            close[short_entry_mask] + params["stop_atr_mult"] * atr[short_entry_mask]
        )
        df.loc[short_entry_mask, "bb_tp_short"] = (
            close[short_entry_mask] - params["tp_atr_mult"] * atr[short_entry_mask]
        )
        signals.iloc[:warmup] = 0.0
        return signals
