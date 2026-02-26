from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_volatility_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_atr_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_atr_mult': ParameterSpec(
                name='keltner_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
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
                default=3.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        volume = df["volume"].values
        # compute sma of volume oscillator
        vol_osc_sma = np.full(n, np.nan)
        fast = int(params.get("volume_oscillator_fast", 12))
        slow = int(params.get("volume_oscillator_slow", 26))
        if fast < n and slow < n and (slow - fast) > 0:
            vol_osc_sma[fast:] = np.convolve(vol_osc, np.ones(fast)/fast, mode='valid')[:n-fast]
        vol_osc_sma = np.nan_to_num(vol_osc_sma)
        # compute median atr over 7 days (assuming 240 bars per day)
        atr_window = 7 * 240
        if atr_window < n:
            atr_median = np.nanmedian(atr[-atr_window:])
        else:
            atr_median = np.nanmedian(atr)
        # detect breakouts
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        # long breakout
        cross_up_upper = (close > indicators['keltner']['upper']) & (prev_close <= np.roll(indicators['keltner']['upper'], 1))
        vol_confirm_long = vol_osc > vol_osc_sma
        atr_filter_long = atr < atr_median
        long_mask = cross_up_upper & vol_confirm_long & atr_filter_long
        # short breakout
        cross_down_lower = (close < indicators['keltner']['lower']) & (prev_close >= np.roll(indicators['keltner']['lower'], 1))
        vol_confirm_short = vol_osc > vol_osc_sma
        atr_filter_short = atr < atr_median
        short_mask = cross_down_lower & vol_confirm_short & atr_filter_short
        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set dynamic stop-loss and take-profit levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # long entries
        entry_long = long_mask
        if np.any(entry_long):
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # short entries
        entry_short = short_mask
        if np.any(entry_short):
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals