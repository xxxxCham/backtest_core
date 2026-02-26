from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_vwap_atr_slope_scalp_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.7,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=20,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.7,
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

        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        vwap = np.nan_to_num(indicators['vwap'])
        atr = np.nan_to_num(indicators['atr'])

        ema_prev = np.roll(ema, 1)
        ema_prev[0] = np.nan

        # helper cross detection
        def cross_down(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # long entry conditions
        long_cond = (
            (close > ema)
            & (close > vwap)
            & (ema > ema_prev)
            & (close > ema + 0.5 * atr)
        )
        long_mask = long_cond

        # short entry conditions
        short_cond = (
            (close < ema)
            & (close < vwap)
            & (ema < ema_prev)
            & (close < ema - 0.5 * atr)
        )
        short_mask = short_cond

        # avoid overlapping long and short
        long_mask &= ~short_mask
        short_mask &= ~long_mask

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit signals: close crosses below vwap or ema
        exit_long_mask = (cross_down(close, vwap) | cross_down(close, ema)) & (signals == 1.0)
        exit_short_mask = (cross_down(close, vwap) | cross_down(close, ema)) & (signals == -1.0)
        signals[exit_long_mask | exit_short_mask] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.1))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.7))

        # long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
