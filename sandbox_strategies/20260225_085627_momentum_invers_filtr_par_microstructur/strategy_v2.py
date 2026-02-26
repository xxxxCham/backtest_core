from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='filtered_momentum_reversal')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'onchain_smoothing', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'euro_session_end': 17,
         'euro_session_start': 8,
         'leverage': 1,
         'onchain_smoothing_period': 20,
         'sl_atr_mult': 1.5,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=25,
                default=14,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'sl_atr_mult': ParameterSpec(
                name='sl_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        williams_r = np.nan_to_num(indicators['williams_r'])
        onchain_smoothing = np.nan_to_num(indicators['onchain_smoothing'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        euro_session_start = int(params.get("euro_session_start", 8))
        euro_session_end = int(params.get("euro_session_end", 17))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        dt_index = pd.to_datetime(df.index)
        cet_hours = dt_index.tz_convert('Europe/Paris').hour
        euro_session_mask = (cet_hours >= euro_session_start) & (cet_hours <= euro_session_end)

        prev_onchain = np.roll(onchain_smoothing, 1)
        prev_onchain[0] = np.nan
        onchain_rising = (onchain_smoothing > prev_onchain) & ~np.isnan(prev_onchain)
        onchain_falling = (onchain_smoothing < prev_onchain) & ~np.isnan(prev_onchain)

        long_entry_mask = (williams_r > -20) & onchain_rising & euro_session_mask
        short_entry_mask = (williams_r < -80) & onchain_falling & euro_session_mask

        long_mask[warmup:] = long_entry_mask[warmup:]
        short_mask[warmup:] = short_entry_mask[warmup:]

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
