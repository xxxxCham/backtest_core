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

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        bb = indicators['bollinger']
        close = df["close"].values
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        ema_50 = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])

        # Compute EMA 50 trend
        ema_50_diff = np.insert(np.diff(ema_50), 0, 0.0)
        ema_trend = ema_50_diff > 0

        # Volume oscillator cross signals
        vol_fast = volume_osc[:params["volume_oscillator_fast"]]
        vol_slow = volume_osc[:params["volume_oscillator_slow"]]
        vol_avg_fast = np.mean(vol_fast) if len(vol_fast) > 0 else 0.0
        vol_avg_slow = np.mean(vol_slow) if len(vol_slow) > 0 else 0.0

        # Vectorized cross-up and cross-down logic for volume oscillator
        prev_vol = np.roll(volume_osc, 1)
        prev_vol[0] = np.nan
        vol_cross_up = (volume_osc > vol_avg_fast) & (prev_vol <= vol_avg_fast)
        vol_cross_down = (volume_osc < vol_avg_fast) & (prev_vol >= vol_avg_fast)

        # Bollinger band touch logic
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        touch_lower = np.abs(close - lower_bb) < (0.0001 * close)
        touch_upper = np.abs(close - upper_bb) < (0.0001 * close)

        # Entry conditions
        long_entry = touch_lower & vol_cross_up & ema_trend
        short_entry = touch_upper & vol_cross_down & (~ema_trend)

        long_mask = long_entry
        short_mask = short_entry

        # Exit conditions
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        # Exit on crossing upper band
        exit_long = close > upper_bb
        exit_short = close < upper_bb

        # Exit on volume decreasing after 4 bars
        vol_decrease = volume_osc < np.roll(volume_osc, 4)
        exit_long = exit_long | (vol_decrease & (np.arange(n) >= 4))
        exit_short = exit_short | (vol_decrease & (np.arange(n) >= 4))

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management with ATR
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute ATR-based SL/TP for long entries
        entry_long_mask = signals == 1.0
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        # Compute ATR-based SL/TP for short entries
        entry_short_mask = signals == -1.0
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals