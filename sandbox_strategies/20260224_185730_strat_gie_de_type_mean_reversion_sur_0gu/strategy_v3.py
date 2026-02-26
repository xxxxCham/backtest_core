from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'volume_oscillator', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volatility_gating_period': 20,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
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
            'volatility_gating_period': ParameterSpec(
                name='volatility_gating_period',
                min_val=10,
                max_val=50,
                default=20,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        bb = indicators['bollinger']
        close = df["close"].values
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        # compute volume oscillator average
        vol_avg = np.nanmean(volume_osc)
        # prepare masks for cross events
        prev_volume_osc = np.roll(volume_osc, 1)
        prev_volume_osc[0] = np.nan
        vol_cross_up = (volume_osc > vol_avg) & (prev_volume_osc <= vol_avg)
        vol_cross_down = (volume_osc < vol_avg) & (prev_volume_osc >= vol_avg)
        # prepare bollinger band values
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        # entry conditions
        close_touches_lower = np.abs(close - lower_bb) < (0.0001 * close)
        close_touches_upper = np.abs(close - upper_bb) < (0.0001 * close)
        ema_cross_up = (ema > np.roll(ema, 1)) & (np.roll(ema, 1) <= np.roll(ema, 2))
        ema_cross_down = (ema < np.roll(ema, 1)) & (np.roll(ema, 1) >= np.roll(ema, 2))
        # long entry: touch lower band, volume crosses up, ema confirms uptrend
        long_entry = close_touches_lower & vol_cross_up & ema_cross_up
        long_mask = long_entry
        # short entry: touch upper band, volume crosses down, ema confirms downtrend
        short_entry = close_touches_upper & vol_cross_down & ema_cross_down
        short_mask = short_entry
        # exit conditions
        exit_long = close > upper_bb
        exit_short = close < lower_bb
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # volatility gating
        vol_gating_period = int(params.get("volatility_gating_period", 20))
        avg_atr = np.nanmean(atr[-vol_gating_period:])
        volatility_gate = atr >= avg_atr
        # ATR-based SL/TP
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals