from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_sma_supertrend_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 1.75,
         'tp_atr_mult': 5.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=100,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        supertrend_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        sma_val = np.nan_to_num(indicators['sma'])

        # Entry conditions
        long_entry = (
            (supertrend_dir == 1)
            & (adx_val > 25)
            & (close > sma_val)
        )
        short_entry = (
            (supertrend_dir == -1)
            & (adx_val > 25)
            & (close < sma_val)
        )

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Exit conditions
        prev_dir = np.roll(supertrend_dir, 1)
        prev_dir[0] = supertrend_dir[0]
        dir_change = supertrend_dir != prev_dir

        # cross_any helper
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_sma = np.roll(sma_val, 1)
        prev_sma[0] = np.nan
        cross_up = (close > sma_val) & (prev_close <= prev_sma)
        cross_down = (close < sma_val) & (prev_close >= prev_sma)
        cross_any = cross_up | cross_down

        exit_mask = dir_change | (adx_val < 20) | cross_any
        signals[exit_mask] = 0.0

        # ATR based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.75))
        tp_mult = float(params.get("tp_atr_mult", 5.5))

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
