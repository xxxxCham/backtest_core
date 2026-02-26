from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_optimized')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 5.5,
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
                default=2.0,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # wrap indicator arrays
        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_vals = np.nan_to_num(indicators['macd']["signal"])
        hist_vals = np.nan_to_num(indicators['macd']["histogram"])
        rsi_vals = np.nan_to_num(indicators['rsi'])
        atr_vals = np.nan_to_num(indicators['atr'])
        close_vals = df["close"].values

        # helper cross functions
        prev_macd = np.roll(macd_vals, 1)
        prev_signal = np.roll(signal_vals, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up = (macd_vals > signal_vals) & (prev_macd <= prev_signal)
        cross_down = (macd_vals < signal_vals) & (prev_macd >= prev_signal)

        # entry conditions
        long_mask = cross_up & (rsi_vals > params["rsi_oversold"]) & (rsi_vals < params["rsi_overbought"])
        short_mask = cross_down & (rsi_vals > params["rsi_oversold"]) & (rsi_vals < params["rsi_overbought"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close_vals[entry_long] - params["stop_atr_mult"] * atr_vals[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close_vals[entry_long] + params["tp_atr_mult"] * atr_vals[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close_vals[entry_short] + params["stop_atr_mult"] * atr_vals[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close_vals[entry_short] - params["tp_atr_mult"] * atr_vals[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
