from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_volume_adjusted')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'volume_oscillator', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 10,
         'volume_oscillator_slow': 30,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
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
        bb = indicators['bollinger']
        close = df["close"].values
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        ema_50 = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        # compute moving average of volume oscillator
        vol_avg = np.nanmean(volume_osc)
        # detect volume oscillator crossing its average
        prev_volume_osc = np.roll(volume_osc, 1)
        prev_volume_osc[0] = np.nan
        vol_cross_up = (volume_osc > vol_avg) & (prev_volume_osc <= vol_avg)
        vol_cross_down = (volume_osc < vol_avg) & (prev_volume_osc >= vol_avg)
        # detect touches of bollinger bands
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        close_touch_lower = np.abs(close - lower_bb) < (0.001 * close)  # approximate touch
        close_touch_upper = np.abs(close - upper_bb) < (0.001 * close)  # approximate touch
        # trend filter using EMA 50
        ema_trend_up = ema_50 > np.roll(ema_50, 1)
        ema_trend_down = ema_50 < np.roll(ema_50, 1)
        # long entry conditions: touch lower band, volume cross up, EMA trend up
        long_entry_cond = close_touch_lower & vol_cross_up & ema_trend_up
        long_mask = long_entry_cond
        # short entry conditions: touch upper band, volume cross down, EMA trend down
        short_entry_cond = close_touch_upper & vol_cross_down & ema_trend_down
        short_mask = short_entry_cond
        # exit conditions
        # exit on crossing upper band or 4 consecutive volume decreases
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        exit_cross_upper = (close > middle_bb) & (prev_close <= middle_bb)
        vol_decrease = volume_osc < prev_volume_osc
        vol_decrease_4_consecutive = (
            vol_decrease &
            np.roll(vol_decrease, 1) &
            np.roll(vol_decrease, 2) &
            np.roll(vol_decrease, 3)
        )
        vol_decrease_4_consecutive[0] = False
        vol_decrease_4_consecutive[1] = False
        vol_decrease_4_consecutive[2] = False
        vol_decrease_4_consecutive[3] = False
        exit_cond = exit_cross_upper | vol_decrease_4_consecutive
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # volatility gating
        avg_atr = np.nanmean(atr)
        volatility_gate = atr > avg_atr
        # apply volatility gate to signals
        signals[~volatility_gate] = 0.0
        # write SL/TP levels into df
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
