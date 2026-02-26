from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_supertrend_adx_atr_v6')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'supertrend', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'atr_period': 14,
            'leverage': 1,
            'stoch_d_period': 3,
            'stoch_k_period': 14,
            'stoch_smooth_k': 3,
            'stop_atr_mult': 1.1,
            'supertrend_atr_period': 10,
            'supertrend_multiplier': 3.0,
            'tp_atr_mult': 2.7,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_k_period': ParameterSpec(
                name='stoch_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stoch_d_period': ParameterSpec(
                name='stoch_d_period',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stoch_smooth_k': ParameterSpec(
                name='stoch_smooth_k',
                min_val=1,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.7,
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
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # --- extract indicator arrays -------------------------------------------------
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"]).astype(float)

        supertrend = indicators['supertrend']
        # ensure float dtype so we can assign np.nan later
        direction = indicators['supertrend']["direction"].astype(float)

        adx_dict = indicators['adx']
        adx_val = np.nan_to_num(adx_dict["adx"]).astype(float)

        atr = np.nan_to_num(indicators['atr']).astype(float)

        close = df["close"].values

        # --- entry logic ---------------------------------------------------------------
        long_entry = (k < 20) & (direction == 1) & (adx_val > 25)
        short_entry = (k > 80) & (direction == -1) & (adx_val > 25)

        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # --- exit logic ----------------------------------------------------------------
        # stochastic crossing 50
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        cross_up = (k > 50) & (prev_k <= 50)
        cross_down = (k < 50) & (prev_k >= 50)
        stoch_cross_50 = cross_up | cross_down

        # supertrend direction change
        prev_dir = np.roll(direction, 1)
        prev_dir[0] = np.nan
        dir_change_long = (direction != 1) & (prev_dir == 1)
        dir_change_short = (direction != -1) & (prev_dir == -1)

        exit_long = (signals == 1.0) & (stoch_cross_50 | dir_change_long)
        exit_short = (signals == -1.0) & (stoch_cross_50 | dir_change_short)

        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # --- ATR‑based stop‑loss / take‑profit ----------------------------------------
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.1))
        tp_mult = float(params.get("tp_atr_mult", 2.7))

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]

        # --- warmup protection ---------------------------------------------------------
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals