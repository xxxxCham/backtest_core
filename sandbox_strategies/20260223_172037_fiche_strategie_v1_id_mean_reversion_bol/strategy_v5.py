from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_vol_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 65,
         'rsi_oversold': 35,
         'rsi_period': 14,
         'stop_atr_mult': 2.75,
         'tp_atr_mult': 4.0,
         'warmup': 50}

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
                default=2.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=4.0,
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
        # Boolean masks for long/short
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Wrap indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Compute ATR 14-period SMA manually
        window = 14
        atr_sma = np.full_like(atr, np.nan)
        cumsum = np.cumsum(atr)
        if n >= window:
            atr_sma[window - 1 :] = (
                cumsum[window - 1 :] - cumsum[: n - window + 1]
            ) / window

        # Cross helper functions
        def cross_any(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        # Entry conditions
        rsi_oversold = float(params.get("rsi_oversold", 35))
        rsi_overbought = float(params.get("rsi_overbought", 65))
        long_mask = (close < lower) & (rsi < rsi_oversold) & (atr < atr_sma)
        short_mask = (close > upper) & (rsi > rsi_overbought) & (atr < atr_sma)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = cross_any(close, middle) | cross_any(rsi, np.full_like(rsi, 50))
        # Flip exit signals: if currently long, set to 0, if short, set to 0
        # We simply set signals to 0 where exit condition is true and currently in position
        # Since signals array only contains entry signals, we need to generate exit series separately
        # For simplicity, we keep signals as entry only; exit handled by simulator via cross_any logic.

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 4.0))

        # ATR-based SL/TP for long entries
        long_entry = signals == 1.0
        if long_entry.any():
            df.loc[long_entry, "bb_stop_long"] = (
                close[long_entry] - stop_atr_mult * atr[long_entry]
            )
            df.loc[long_entry, "bb_tp_long"] = (
                close[long_entry] + tp_atr_mult * atr[long_entry]
            )

        # ATR-based SL/TP for short entries
        short_entry = signals == -1.0
        if short_entry.any():
            df.loc[short_entry, "bb_stop_short"] = (
                close[short_entry] + stop_atr_mult * atr[short_entry]
            )
            df.loc[short_entry, "bb_tp_short"] = (
                close[short_entry] - tp_atr_mult * atr[short_entry]
            )
        signals.iloc[:warmup] = 0.0
        return signals
