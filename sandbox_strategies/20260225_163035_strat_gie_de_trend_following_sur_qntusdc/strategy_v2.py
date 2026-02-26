from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_ema_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_long_period': 50,
         'ema_short_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 3.8,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
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
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.8,
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
        atr = np.nan_to_num(indicators['atr'])

        # compute EMAs
        ema_short = pd.Series(close).ewm(span=params["ema_short_period"], adjust=False).mean().values
        ema_long = pd.Series(close).ewm(span=params["ema_long_period"], adjust=False).mean().values

        # vortex indicator
        vx = indicators['vortex']
        vip = np.nan_to_num(vx["vi_plus"])
        vim = np.nan_to_num(vx["vi_minus"])

        # cross helpers
        prev_vip = np.roll(vip, 1); prev_vip[0] = np.nan
        prev_vim = np.roll(vim, 1); prev_vim[0] = np.nan
        cross_up_v = (vip > vim) & (prev_vip <= prev_vim)
        cross_down_v = (vip < vim) & (prev_vip >= prev_vim)

        # long entry: vortex bullish + EMA filter
        long_mask = (vip > vim) & (close > ema_short) & (ema_short > ema_long)
        # short entry: vortex bearish + EMA filter
        short_mask = (vim > vip) & (close < ema_short) & (ema_short < ema_long)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[signals == 1.0, "bb_stop_long"] = close[signals == 1.0] - stop_atr_mult * atr[signals == 1.0]
        df.loc[signals == 1.0, "bb_tp_long"] = close[signals == 1.0] + tp_atr_mult * atr[signals == 1.0]
        df.loc[signals == -1.0, "bb_stop_short"] = close[signals == -1.0] + stop_atr_mult * atr[signals == -1.0]
        df.loc[signals == -1.0, "bb_tp_short"] = close[signals == -1.0] - tp_atr_mult * atr[signals == -1.0]
        signals.iloc[:warmup] = 0.0
        return signals
