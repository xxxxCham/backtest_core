from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rune_30m_ema_vwap_rsi_scalp')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'vwap', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_long_period': 50,
         'ema_short_period': 20,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_short_period': ParameterSpec(
                name='ema_short_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_long_period': ParameterSpec(
                name='ema_long_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.2,
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

        close = df["close"].values

        ema_raw = indicators['ema']
        if isinstance(ema_raw, dict):
            ema_short = np.nan_to_num(ema_raw.get("short", np.zeros(n)))
            ema_long = np.nan_to_num(ema_raw.get("long", np.zeros(n)))
        else:
            ema_short = ema_long = np.nan_to_num(ema_raw)

        vwap = np.nan_to_num(indicators['vwap'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        def cross_up(x, y):
            px = np.roll(x, 1); px[0] = np.nan
            py = np.roll(y, 1); py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x, y):
            px = np.roll(x, 1); px[0] = np.nan
            py = np.roll(y, 1); py[0] = np.nan
            return (x < y) & (px >= py)

        long_mask = (
            cross_up(close, ema_short)
            & (ema_short > ema_long)
            & (close > vwap)
            & (rsi < params["rsi_overbought"])
        )

        short_mask = (
            cross_down(close, ema_short)
            & (ema_short < ema_long)
            & (close < vwap)
            & (rsi > params["rsi_oversold"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        prev_signals = np.roll(signals.values, 1)
        prev_signals[0] = 0.0

        long_exit_mask = (
            (prev_signals == 1.0)
            & ((cross_down(close, ema_short)) | (rsi > params["rsi_overbought"]))
        )
        short_exit_mask = (
            (prev_signals == -1.0)
            & ((cross_up(close, ema_short)) | (rsi < params["rsi_oversold"]))
        )

        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
