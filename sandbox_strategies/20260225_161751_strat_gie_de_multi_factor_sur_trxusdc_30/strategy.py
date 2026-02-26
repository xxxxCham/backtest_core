from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_macd_adx_atr_triple_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 25,
         'leverage': 1,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 5.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.9,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=10,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        supertrend = np.nan_to_num(indicators['supertrend']["supertrend"])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        adx_threshold = params.get("adx_threshold", 25.0)
        stop_atr_mult = params.get("stop_atr_mult", 2.2)
        tp_atr_mult = params.get("tp_atr_mult", 5.9)

        # helper cross functions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_supertrend = np.roll(supertrend, 1)
        prev_supertrend[0] = np.nan

        cross_down = (close < supertrend) & (prev_close >= prev_supertrend)
        cross_up = (close > supertrend) & (prev_close <= prev_supertrend)

        # Entry conditions
        long_mask = (close > supertrend) & (macd_hist > 0) & (adx > adx_threshold)
        short_mask = (close < supertrend) & (macd_hist < 0) & (adx > adx_threshold)

        # Exit conditions
        exit_long_mask = cross_down | (macd_hist < 0) | (adx < 20)
        exit_short_mask = cross_up | (macd_hist > 0) | (adx < 20)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask | exit_short_mask] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set SL/TP only on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
