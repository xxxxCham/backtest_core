from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_ema_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.75,
         'tp_atr_mult': 2.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
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
        # Boolean masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Wrap indicator arrays
        close = df["close"].values
        direction = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        ema_val = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (direction == 1.0) & (adx_val > 25.0) & (close > ema_val)
        short_mask = (direction == -1.0) & (adx_val > 25.0) & (close < ema_val)

        # Avoid duplicate consecutive signals
        prev_signals = np.roll(signals.values, 1)
        prev_signals[0] = 0.0
        new_long = long_mask & (prev_signals != 1.0)
        new_short = short_mask & (prev_signals != -1.0)
        signals[new_long] = 1.0
        signals[new_short] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute SL/TP only on entry bars
        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        df.loc[new_long, "bb_stop_long"] = close[new_long] - stop_atr_mult * atr[new_long]
        df.loc[new_long, "bb_tp_long"] = close[new_long] + tp_atr_mult * atr[new_long]
        df.loc[new_short, "bb_stop_short"] = close[new_short] + stop_atr_mult * atr[new_short]
        df.loc[new_short, "bb_tp_short"] = close[new_short] - tp_atr_mult * atr[new_short]
        signals.iloc[:warmup] = 0.0
        return signals
