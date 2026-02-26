from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='frontusdc_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50,
         'williams_r_overbought': -20,
         'williams_r_oversold': -80,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
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
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        close = np.nan_to_num(df["close"].values)
        ema_fast = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])
        # EMA slow for exit filter
        ema_slow = np.nan_to_num(indicators['ema'])
        # compute EMA of volume oscillator
        vol_ema = np.full(n, np.nan)
        vol_ema[params["ema_fast"] - 1 :] = np.convolve(volume_oscillator, np.ones(params["ema_fast"]) / params["ema_fast"], mode="valid")
        vol_ema = np.nan_to_num(vol_ema)
        # entry conditions
        # long entry: close touches lower EMA band, volume oscillator > EMA of volume oscillator, williams_r < -80
        lower_ema = ema_fast - 2 * atr  # approximate lower band
        close_touches_lower = np.abs(close - lower_ema) < 0.001 * close
        volume_condition = volume_oscillator > vol_ema
        oversold_condition = williams_r < params["williams_r_oversold"]
        long_entry = close_touches_lower & volume_condition & oversold_condition
        # short entry: close touches upper EMA band, volume oscillator > EMA of volume oscillator, williams_r > -80
        upper_ema = ema_fast + 2 * atr  # approximate upper band
        close_touches_upper = np.abs(close - upper_ema) < 0.001 * close
        overbought_condition = williams_r > params["williams_r_oversold"]
        short_entry = close_touches_upper & volume_condition & overbought_condition
        # exit conditions
        # exit long: close crosses below EMA 50
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_slow[0] = np.nan
        exit_long = (close < ema_slow) & (prev_close >= prev_ema_slow)
        # exit short: close crosses above EMA 50
        exit_short = (close > ema_slow) & (prev_close <= prev_ema_slow)
        # combine signals
        long_mask = long_entry
        short_mask = short_entry
        # apply exit conditions
        # for longs
        long_exit_mask = np.zeros(n, dtype=bool)
        prev_exit_long = np.roll(exit_long, 1)
        prev_exit_long[0] = False
        long_exit_mask = exit_long | (williams_r > params["williams_r_overbought"])
        # for shorts
        short_exit_mask = np.zeros(n, dtype=bool)
        prev_exit_short = np.roll(exit_short, 1)
        prev_exit_short[0] = False
        short_exit_mask = exit_short | (williams_r < params["williams_r_overbought"])
        # ensure no overlapping positions
        long_mask = long_mask & ~short_mask
        short_mask = short_mask & ~long_mask
        # update signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # long stop-loss and take-profit
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # short stop-loss and take-profit
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
