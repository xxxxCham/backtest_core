from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='filtered_reversal_european_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'onchain_smoothing', 'atr', 'momentum']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'european_hours_end': 16,
         'european_hours_start': 8,
         'leverage': 1,
         'momentum_period': 10,
         'onchain_smoothing_period': 5,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 2.0,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
            'onchain_smoothing_period': ParameterSpec(
                name='onchain_smoothing_period',
                min_val=3,
                max_val=10,
                default=5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        williams_r = np.nan_to_num(indicators['williams_r'])
        onchain_smoothing = np.nan_to_num(indicators['onchain_smoothing'])
        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])

        close = df["close"].values
        hours = df.index.hour

        european_start = params.get("european_hours_start", 8)
        european_end = params.get("european_hours_end", 16)
        session_mask = (hours >= european_start) & (hours <= european_end)

        prev_williams_r = np.roll(williams_r, 1)
        prev_onchain_smoothing = np.roll(onchain_smoothing, 1)
        prev_williams_r[0] = np.nan
        prev_onchain_smoothing[0] = np.nan

        williams_r_cross_above_20 = (williams_r > -20) & (prev_williams_r <= -20)
        williams_r_cross_below_80 = (williams_r < -80) & (prev_williams_r >= -80)

        onchain_up = onchain_smoothing > prev_onchain_smoothing
        onchain_down = onchain_smoothing < prev_onchain_smoothing

        long_conditions = (
            williams_r_cross_above_20 & 
            onchain_up & 
            (momentum > 0) & 
            session_mask
        )

        short_conditions = (
            williams_r_cross_below_80 & 
            onchain_down & 
            (momentum < 0) & 
            session_mask
        )

        long_mask = long_conditions
        short_mask = short_conditions

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        long_entry_mask = (signals == 1.0)
        short_entry_mask = (signals == -1.0)

        df.loc[long_entry_mask, "bb_stop_long"] = close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        df.loc[long_entry_mask, "bb_tp_long"] = close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]

        df.loc[short_entry_mask, "bb_stop_short"] = close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        df.loc[short_entry_mask, "bb_tp_short"] = close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
