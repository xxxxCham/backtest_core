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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # prepare arrays for cross detection
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_obv = np.roll(obv, 1)
        prev_volume_osc = np.roll(volume_osc, 1)
        prev_volume_osc_2 = np.roll(volume_osc, 2)
        prev_volume_osc_3 = np.roll(volume_osc, 3)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        prev_obv[0] = np.nan
        prev_volume_osc[0] = np.nan
        prev_volume_osc_2[0] = np.nan
        prev_volume_osc_3[0] = np.nan
        # define entry conditions
        ema_cross_up = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)
        obv_up = obv > prev_obv
        volume_osc_up = volume_osc > prev_volume_osc
        volume_osc_up_2 = volume_osc > prev_volume_osc_2
        volume_osc_up_3 = volume_osc > prev_volume_osc_3
        # long entry
        long_entry = ema_cross_up & obv_up & volume_osc_up & volume_osc_up_2 & volume_osc_up_3
        long_mask = long_entry
        # short entry
        short_entry = ema_cross_down & ~obv_up & volume_osc_up & volume_osc_up_2 & volume_osc_up_3
        short_mask = short_entry
        # exit conditions
        ema_cross_down_exit = ema_cross_down
        volume_osc_down = volume_osc < prev_volume_osc
        # exit long
        long_exit = ema_cross_down_exit | volume_osc_down
        # exit short
        short_exit = ema_cross_up | volume_osc_down
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # handle exits
        exit_long_mask = long_exit & (np.roll(signals, 1) == 1.0)
        exit_short_mask = short_exit & (np.roll(signals, 1) == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
        close = df["close"].values
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
