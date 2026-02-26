from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_sur_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'obv_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=2.0,
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

        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])

        # EMA arrays
        ema_50 = ema_fast
        ema_200 = ema_slow

        # Volume oscillators
        vo_fast = volume_oscillator

        # Rolling volumes for OBV and volume oscillator
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        prev_vo = np.roll(vo_fast, 1)
        prev_vo[0] = np.nan

        prev_vo_2 = np.roll(vo_fast, 2)
        prev_vo_2[0] = np.nan
        prev_vo_2[1] = np.nan

        prev_vo_3 = np.roll(vo_fast, 3)
        prev_vo_3[0] = np.nan
        prev_vo_3[1] = np.nan
        prev_vo_3[2] = np.nan

        # Entry conditions
        ema_long_condition = (ema_50 > ema_200)
        obv_long_condition = (obv > prev_obv)
        vo_long_condition = (vo_fast > prev_vo)
        vo_trend_condition = (prev_vo > prev_vo_2) & (prev_vo_2 > prev_vo_3)

        long_condition = ema_long_condition & obv_long_condition & vo_long_condition & vo_trend_condition

        # Exit condition
        ema_exit_condition = (ema_50 < ema_200)
        vo_exit_condition = (vo_fast < prev_vo)
        exit_condition = ema_exit_condition | vo_exit_condition

        # Set long mask
        long_mask = long_condition
        signals[long_mask] = 1.0

        # Set short mask
        short_mask = (ema_50 < ema_200) & ~long_mask
        signals[short_mask] = -1.0

        # Set warmup
        signals.iloc[:warmup] = 0.0

        # Risk management
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan

        entry_mask = (signals == 1.0)
        close = df["close"].values

        df.loc[entry_mask, "bb_stop_long"] = close[entry_mask] - stop_atr_mult * atr[entry_mask]
        df.loc[entry_mask, "bb_tp_long"] = close[entry_mask] + tp_atr_mult * atr[entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals