from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_5m_ema_bollinger_atr_scalp')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_fast_period': 9,
         'ema_slow_period': 21,
         'leverage': 1,
         'stop_atr_mult': 2.3,
         'tp_atr_mult': 4.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast_period': ParameterSpec(
                name='ema_fast_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ema_slow_period': ParameterSpec(
                name='ema_slow_period',
                min_val=15,
                max_val=50,
                default=21,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.5,
                max_val=3.0,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
                default=4.6,
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
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays with NaN handling
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])

        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_entry = (ema > middle) & (close > upper)
        short_entry = (ema < middle) & (close < lower)

        # Populate masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop‑loss / take‑profit columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 2.3))
        tp_mult = float(params.get("tp_atr_mult", 4.6))

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        # Exit on EMA crossing opposite side of Bollinger middle (optional)
        prev_ema = np.roll(ema, 1)
        prev_middle = np.roll(middle, 1)
        prev_ema[0] = np.nan
        prev_middle[0] = np.nan

        cross_down = (ema < middle) & (prev_ema >= prev_middle)   # EMA crosses below middle
        cross_up = (ema > middle) & (prev_ema <= prev_middle)   # EMA crosses above middle

        # Clear signals on exit crosses (ensure flat position)
        exit_long = cross_down & long_mask
        exit_short = cross_up & short_mask
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
