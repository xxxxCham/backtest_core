from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_rsi')

    @property
    def required_indicators(self) -> List[str]:
        # Include ATR because the strategy uses it for stop‑loss and take‑profit levels
        return ['donchian', 'adx', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_period': 14,
            'stop_atr_mult': 1.75,
            'tp_atr_mult': 5.5,
            'warmup': 25
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays with NaN handling
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        donch = indicators['donchian']
        donch_upper = np.nan_to_num(donch["upper"])
        donch_lower = np.nan_to_num(donch["lower"])
        donch_middle = np.nan_to_num(donch["middle"])

        # Long entry condition
        long_cond = (
            (close > donch_upper)
            & (adx_val > 20)
            & (rsi > 50)
        )
        long_mask[long_cond] = True

        # Short entry condition
        short_cond = (
            (close < donch_lower)
            & (adx_val > 20)
            & (rsi < 50)
        )
        short_mask[short_cond] = True

        # Exit condition: close crosses donch_middle OR adx < 15
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_up = (close > donch_middle) & (prev_close <= np.roll(donch_middle, 1))
        cross_down = (close < donch_middle) & (prev_close >= np.roll(donch_middle, 1))
        cross_any = cross_up | cross_down
        exit_cond = cross_any | (adx_val < 15)

        # Apply masks
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_cond] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Write ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.75))
        tp_mult = float(params.get("tp_atr_mult", 5.5))

        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_mult * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_mult * atr[long_entries]

        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_mult * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_mult * atr[short_entries]

        # Final warmup protection
        signals.iloc[:warmup] = 0.0
        return signals