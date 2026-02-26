from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='paxg_usdc_mean_reversion_williams_obv_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 4.1,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
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
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=8.0,
                default=4.1,
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

        # unwrap indicators
        williams = np.nan_to_num(indicators['williams_r'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # compute previous values for cross detection
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan
        williams_prev = np.roll(williams, 1)
        williams_prev[0] = np.nan

        # entry conditions
        long_mask = (williams < -80.0) & (obv > obv_prev)
        short_mask = (williams > -20.0) & (obv < obv_prev)

        # exit conditions
        williams_cross_up = (williams > -50.0) & (williams_prev <= -50.0)
        # obv trend opposite of entry
        obv_opposite_long = obv < obv_prev
        obv_opposite_short = obv > obv_prev
        exit_long_mask = williams_cross_up | obv_opposite_long
        exit_short_mask = williams_cross_up | obv_opposite_short

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # write ATR based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.7))
        tp_mult = float(params.get("tp_atr_mult", 4.1))

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_mult * atr[long_entry]
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
