from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_atr_adx_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 4.8,
         'trailing_atr_mult': 2.3,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.8,
                param_type='float',
                step=0.1,
            ),
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
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

        # Extract indicator arrays
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])

        atr = np.nan_to_num(indicators['atr'])
        adx_d = indicators['adx']
        adx = np.nan_to_num(adx_d["adx"])

        close = df["close"].values

        # Parameter values
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 4.8)
        trailing_atr_mult = params.get("trailing_atr_mult", 2.3)
        adx_entry_threshold = params.get("adx_entry_threshold", 25.0)
        adx_exit_threshold = params.get("adx_exit_threshold", 20.0)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Previous signal state for entry/exit logic
        prev_signal = np.roll(signals.values, 1)
        prev_signal[0] = 0

        # Long entry: close > upper AND adx > entry threshold AND flat
        long_entry = (
            (prev_signal == 0)
            & (close > upper)
            & (adx > adx_entry_threshold)
        )
        long_mask = long_entry

        # Short entry: close < lower AND adx > entry threshold AND flat
        short_entry = (
            (prev_signal == 0)
            & (close < lower)
            & (adx > adx_entry_threshold)
        )
        short_mask = short_entry

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Long exit: close < middle OR adx < exit threshold
        long_exit = (
            (prev_signal == 1)
            & ((close < middle) | (adx < adx_exit_threshold))
        )
        signals[long_exit] = 0.0

        # Short exit: close > middle OR adx < exit threshold
        short_exit = (
            (prev_signal == -1)
            & ((close > middle) | (adx < adx_exit_threshold))
        )
        signals[short_exit] = 0.0

        # Write ATR-based SL/TP columns on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        # Optional: trailing stop columns could be added similarly if simulator supports
        signals.iloc[:warmup] = 0.0
        return signals
