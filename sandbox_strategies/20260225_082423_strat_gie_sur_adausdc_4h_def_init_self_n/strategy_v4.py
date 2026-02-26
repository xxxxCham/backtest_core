from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_momentum_reversal')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'momentum', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'momentum_period': 10,
         'stop_atr_mult': 2.0,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=7,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=2,
                max_val=5,
                default=3,
                param_type='float',
                step=0.1,
            ),
            'momentum_period': ParameterSpec(
                name='momentum_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=3.0,
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
        st = indicators['supertrend']
        supertrend = np.nan_to_num(st["supertrend"])
        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])

        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        prev_close = np.roll(close, 1)
        prev_supertrend = np.roll(supertrend, 1)
        prev_close[0] = np.nan
        prev_supertrend[0] = np.nan

        long_entry = (close > supertrend) & (momentum > 0)
        short_entry = (close < supertrend) & (momentum < 0)

        long_exit = (close < supertrend) & (prev_close >= prev_supertrend)
        short_exit = (close > supertrend) & (prev_close <= prev_supertrend)

        long_mask = long_entry & ~long_exit
        short_mask = short_entry & ~short_exit

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        long_entry_bars = (signals == 1.0) & (np.roll(signals, 1) != 1.0)
        short_entry_bars = (signals == -1.0) & (np.roll(signals, 1) != -1.0)

        df.loc[long_entry_bars, "bb_stop_long"] = close[long_entry_bars] - stop_atr_mult * atr[long_entry_bars]
        df.loc[long_entry_bars, "bb_tp_long"] = close[long_entry_bars] + tp_atr_mult * atr[long_entry_bars]
        df.loc[short_entry_bars, "bb_stop_short"] = close[short_entry_bars] + stop_atr_mult * atr[short_entry_bars]
        df.loc[short_entry_bars, "bb_tp_short"] = close[short_entry_bars] - tp_atr_mult * atr[short_entry_bars]
        signals.iloc[:warmup] = 0.0
        return signals
