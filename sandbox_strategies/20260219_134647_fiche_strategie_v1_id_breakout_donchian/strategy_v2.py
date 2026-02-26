from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='donchian_adx_rsi_breakout')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is required for stop‑and‑take calculations
        return ['donchian', 'adx', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 2.5, 'warmup': 30}

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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Boolean masks initialization
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays with np.nan_to_num
        close = df["close"].values
        donchian = indicators['donchian']
        upper = np.nan_to_num(indicators['donchian']["upper"])
        lower = np.nan_to_num(indicators['donchian']["lower"])
        middle = np.nan_to_num(indicators['donchian']["middle"])

        adx_vals = np.nan_to_num(indicators['adx']["adx"])
        rsi_vals = np.nan_to_num(indicators['rsi'])
        atr_vals = np.nan_to_num(indicators['atr'])

        # Long entry condition
        long_mask = (close > upper) & (rsi_vals > 50.0) & (adx_vals > 25.0)
        # Short entry condition
        short_mask = (close < lower) & (rsi_vals < 50.0) & (adx_vals > 25.0)

        # Assign signals for entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit condition: cross_any(close, middle) or adx < 20
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        cross_up = (close > middle) & (prev_close <= prev_middle)
        cross_down = (close < middle) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down
        exit_mask = cross_any | (adx_vals < 20.0)

        # Apply exit signals (flatten positions)
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Write ATR-based SL/TP levels for long and short entries
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        # Long stop/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr_vals[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr_vals[long_mask]

        # Short stop/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr_vals[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr_vals[short_mask]

        return signals