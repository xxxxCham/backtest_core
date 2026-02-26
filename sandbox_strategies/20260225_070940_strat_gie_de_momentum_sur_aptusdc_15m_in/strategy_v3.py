from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='roc_macd_atr_momentum_15m_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_vol_thr': 0.0005,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'roc_entry_thr': 0.5,
         'roc_exit_thr': 0.2,
         'roc_period': 9,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=9,
                param_type='int',
                step=1,
            ),
            'roc_entry_thr': ParameterSpec(
                name='roc_entry_thr',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'roc_exit_thr': ParameterSpec(
                name='roc_exit_thr',
                min_val=0.05,
                max_val=1.0,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=8,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=20,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=15,
                default=9,
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
                min_val=0.8,
                max_val=5.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'atr_vol_thr': ParameterSpec(
                name='atr_vol_thr',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
                default=50,
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
        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # Extract indicators with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        macd_dict = indicators['macd']
        macd_hist = np.nan_to_num(macd_dict["histogram"])
        atr = np.nan_to_num(indicators['atr'])

        # Price series
        close = df["close"].values

        # Parameter shortcuts
        roc_entry_thr = float(params.get("roc_entry_thr", 0.5))
        roc_exit_thr = float(params.get("roc_exit_thr", 0.2))
        atr_vol_thr = float(params.get("atr_vol_thr", 0.0005))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.2))

        # Entry conditions
        long_entry = (roc > roc_entry_thr) & (macd_hist > 0) & (atr > atr_vol_thr)
        short_entry = (roc < -roc_entry_thr) & (macd_hist < 0) & (atr > atr_vol_thr)

        # Ensure masks are proper length
        assert len(long_entry) == n
        assert len(short_entry) == n

        # Apply entry signals
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Prepare SL/TP columns (initialize with NaN)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute SL/TP for long entries
        if long_entry.any():
            entry_price_long = close[long_entry]
            atr_long = atr[long_entry]
            df.loc[long_entry, "bb_stop_long"] = entry_price_long - stop_atr_mult * atr_long
            df.loc[long_entry, "bb_tp_long"] = entry_price_long + tp_atr_mult * atr_long

        # Compute SL/TP for short entries
        if short_entry.any():
            entry_price_short = close[short_entry]
            atr_short = atr[short_entry]
            df.loc[short_entry, "bb_stop_short"] = entry_price_short + stop_atr_mult * atr_short
            df.loc[short_entry, "bb_tp_short"] = entry_price_short - tp_atr_mult * atr_short

        # Exit logic (flat signal) – override any entry if exit condition met
        # Detect MACD histogram crossing zero
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        cross_up = (macd_hist > 0) & (prev_hist <= 0)
        cross_down = (macd_hist < 0) & (prev_hist >= 0)
        macd_cross = cross_up | cross_down

        long_exit = (roc < roc_exit_thr) | macd_cross
        short_exit = (roc > -roc_exit_thr) | macd_cross

        # Ensure we don't keep a position after an exit on the same bar
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
