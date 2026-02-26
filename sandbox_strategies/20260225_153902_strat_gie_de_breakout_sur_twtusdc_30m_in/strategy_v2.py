from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='donchian_atr_adx_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.3, 'tp_atr_mult': 2.2, 'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=20,
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

        # unpack indicators with np.nan_to_num
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        middle = np.nan_to_num(dc["middle"])
        lower = np.nan_to_num(dc["lower"])
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # cross helper
        prev_upper = np.roll(upper, 1)
        prev_lower = np.roll(lower, 1)
        prev_middle = np.roll(middle, 1)
        prev_upper[0] = np.nan
        prev_lower[0] = np.nan
        prev_middle[0] = np.nan

        # long entry: close crosses above upper AND adx > 25
        long_cross = (close > upper) & (np.roll(close, 1) <= prev_upper)
        long_mask = long_cross & (adx_val > 25)

        # short entry: close crosses below lower AND adx > 25
        short_cross = (close < lower) & (np.roll(close, 1) >= prev_lower)
        short_mask = short_cross & (adx_val > 25)

        # apply masks
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # write SL/TP columns for ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.3))
        tp_mult = float(params.get("tp_atr_mult", 2.2))

        # long entry levels
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr[entry_long]

        # short entry levels
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
