from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 19,
         'atr_period': 14,
         'donchian_period': 25,
         'leverage': 1,
         'stop_atr_mult': 3.0,
         'tp_atr_mult': 3.5,
         'warmup': 25}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=19,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=10,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
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
        # Boolean masks for long/short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        close = df["close"].values
        # Donchian bands
        dc = indicators['donchian']
        indicators['donchian']['upper'] = np.nan_to_num(dc["upper"])
        indicators['donchian']['lower'] = np.nan_to_num(dc["lower"])
        indicators['donchian']['middle'] = np.nan_to_num(dc["middle"])
        # ADX
        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        # ATR
        atr_vals = np.nan_to_num(indicators['atr'])

        # Entry logic
        long_mask = (close > indicators['donchian']['upper']) & (adx_vals > 30)
        short_mask = (close < indicators['donchian']['lower']) & (adx_vals > 30)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic: cross of close and Donchian middle or ADX below 15
        prev_close = np.roll(close, 1)
        prev_mid = np.roll(indicators['donchian']['middle'], 1)
        prev_close[0] = np.nan
        prev_mid[0] = np.nan
        cross_up = (close > indicators['donchian']['middle']) & (prev_close <= prev_mid)
        cross_down = (close < indicators['donchian']['middle']) & (prev_close >= prev_mid)
        cross_any = cross_up | cross_down
        exit_mask = cross_any | (adx_vals < 15)

        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 3.0)
        tp_mult = params.get("tp_atr_mult", 3.5)

        # Long entry SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr_vals[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr_vals[long_mask]

        # Short entry SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr_vals[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr_vals[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
