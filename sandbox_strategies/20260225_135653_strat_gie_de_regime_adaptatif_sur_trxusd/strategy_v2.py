from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trxusdc_30m_regime_adaptive_atr_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'donchian']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'atr_threshold': 0.0005,
         'donchian_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.5,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 2.0,
         'tp_atr_mult_trend': 4.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
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
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0001,
                max_val=1.0,
                default=0.0005,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=1.0,
                max_val=6.0,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        middle = np.nan_to_num(dc["middle"])
        lower = np.nan_to_num(dc["lower"])

        # Helper cross functions
        def cross_up(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Entry conditions
        long_cond = cross_up(close, upper) & (adx_val > 25) & (atr > params["atr_threshold"])
        short_cond = cross_down(close, lower) & (adx_val > 25) & (atr > params["atr_threshold"])
        long_mask = long_cond
        short_mask = short_cond

        # Exit conditions
        exit_long = cross_down(close, middle) | (adx_val < 25)
        exit_short = cross_up(close, middle) | (adx_val < 25)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Optionally clear flat signals on exit (not strictly required)
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry levels
        if long_mask.any():
            entry_idx = np.where(long_mask)[0]
            entry_close = close[entry_idx]
            entry_atr = atr[entry_idx]
            entry_adx = adx_val[entry_idx]
            tp_mult = np.where(entry_adx > 25, params["tp_atr_mult_trend"], params["tp_atr_mult_range"])
            df.loc[entry_idx, "bb_stop_long"] = entry_close - params["stop_atr_mult"] * entry_atr
            df.loc[entry_idx, "bb_tp_long"] = entry_close + tp_mult * entry_atr

        # Short entry levels
        if short_mask.any():
            entry_idx = np.where(short_mask)[0]
            entry_close = close[entry_idx]
            entry_atr = atr[entry_idx]
            entry_adx = adx_val[entry_idx]
            tp_mult = np.where(entry_adx > 25, params["tp_atr_mult_trend"], params["tp_atr_mult_range"])
            df.loc[entry_idx, "bb_stop_short"] = entry_close + params["stop_atr_mult"] * entry_atr
            df.loc[entry_idx, "bb_tp_short"] = entry_close - tp_mult * entry_atr
        signals.iloc[:warmup] = 0.0
        return signals
