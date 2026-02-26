from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_ema_adx')

    @property
    def required_indicators(self) -> List[str]:
        # ATR is required for risk management
        return ['supertrend', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.25, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.25,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Prepare indicator arrays as floats to allow NaN handling
        close = np.nan_to_num(df["close"].values, nan=0.0)
        ema = np.array(indicators['ema'], dtype=float)
        st_dir = np.array(indicators['supertrend']["direction"], dtype=float)
        adx_val = np.array(indicators['adx']["adx"], dtype=float)
        atr = np.array(indicators['atr'], dtype=float)

        # Long / short entry conditions
        long_mask = (st_dir == 1) & (adx_val > 25) & (close > ema)
        short_mask = (st_dir == -1) & (adx_val > 25) & (close < ema)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        prev_dir = np.roll(st_dir, 1)
        prev_dir[0] = np.nan
        direction_change = (st_dir != prev_dir) & (~np.isnan(prev_dir))

        cross_down = (close < ema) & (np.roll(close, 1) >= np.roll(ema, 1))
        cross_down[0] = False

        exit_mask = direction_change | cross_down | (adx_val < 20)

        # Apply exit masks based on prior position
        long_exit = exit_mask & (signals.shift(1, fill_value=0) == 1.0)
        short_exit = exit_mask & (signals.shift(1, fill_value=0) == -1.0)

        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # ATR-based stop‑loss / take‑profit
        stop_atr_mult = float(params.get("stop_atr_mult", 2.25))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]

        # Ensure warmup period is flat
        signals.iloc[:warmup] = 0.0
        return signals