from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_volume_breakout_ltcusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
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
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
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
                min_val=1.0,
                max_val=6.0,
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
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # compute EMA for OBV and volume oscillator
        obv_ema = np.roll(obv, params["obv_period"])
        obv_ema[params["obv_period"] - 1 :] = np.nan
        volume_oscillator_ema = np.roll(volume_oscillator, params["volume_oscillator_fast"])
        volume_oscillator_ema[params["volume_oscillator_fast"] - 1 :] = np.nan
        # compute previous close for crossover
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        # long entry conditions
        cross_above = (close > ema) & (prev_close <= ema)
        obv_positive = obv > obv_ema
        vol_positive = volume_oscillator > 0
        vol_above_ema = volume_oscillator > volume_oscillator_ema
        long_entry = cross_above & obv_positive & vol_positive & vol_above_ema
        long_mask = long_entry
        # short entry conditions
        cross_below = (close < ema) & (prev_close >= ema)
        obv_negative = obv < obv_ema
        vol_negative = volume_oscillator < 0
        vol_below_ema = volume_oscillator < volume_oscillator_ema
        short_entry = cross_below & obv_negative & vol_negative & vol_below_ema
        short_mask = short_entry
        # exit conditions
        ema_cross = (close < ema) & (prev_close >= ema)
        vol_cross = (volume_oscillator < 0) & (np.roll(volume_oscillator, 1) >= 0)
        long_exit = ema_cross | vol_cross
        short_exit = (close > ema) & (prev_close <= ema)
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP levels for long entries
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # set SL/TP levels for short entries
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[:, "bb_stop_short"] = np.nan
            df.loc[:, "bb_tp_short"] = np.nan
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
