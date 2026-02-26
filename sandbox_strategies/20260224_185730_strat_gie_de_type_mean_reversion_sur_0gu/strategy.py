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
         'volatility_filter_period': 20,
         'volume_oscillator_long': 20,
         'volume_oscillator_short': 5,
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
            'volatility_filter_period': ParameterSpec(
                name='volatility_filter_period',
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

        # Extract indicators
        bb = indicators['bollinger']
        close = df["close"].values
        ema_50 = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])

        # Compute volume oscillator averages
        vol_long = params["volume_oscillator_long"]
        vol_short = params["volume_oscillator_short"]
        vol_avg_long = np.convolve(volume_oscillator, np.ones(vol_long)/vol_long, mode='valid')
        vol_avg_long = np.pad(vol_avg_long, (vol_long - 1, 0), constant_values=np.nan)
        vol_avg_short = np.convolve(volume_oscillator, np.ones(vol_short)/vol_short, mode='valid')
        vol_avg_short = np.pad(vol_avg_short, (vol_short - 1, 0), constant_values=np.nan)

        # Cross up/down helpers
        prev_volume_oscillator = np.roll(volume_oscillator, 1)
        prev_volume_oscillator[0] = np.nan
        vol_cross_up_long = (volume_oscillator > vol_avg_long) & (prev_volume_oscillator <= vol_avg_long)
        vol_cross_down_short = (volume_oscillator < vol_avg_short) & (prev_volume_oscillator >= vol_avg_short)

        # Bollinger bands
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])

        # Entry conditions
        # Long entry: close touches lower band, volume crosses above average, ema 50 rising
        close_touches_lower = np.abs(close - indicators['bollinger']['lower']) < (0.001 * close)
        ema_50_rising = np.diff(ema_50) > 0
        ema_50_rising = np.insert(ema_50_rising, 0, False)
        long_entry_cond = close_touches_lower & vol_cross_up_long & ema_50_rising

        # Short entry: close touches upper band, volume crosses above average, ema 50 falling
        close_touches_upper = np.abs(close - indicators['bollinger']['upper']) < (0.001 * close)
        ema_50_falling = np.diff(ema_50) < 0
        ema_50_falling = np.insert(ema_50_falling, 0, False)
        short_entry_cond = close_touches_upper & vol_cross_up_long & ema_50_falling

        # Exit conditions
        # Exit long: close crosses above middle band OR (close crosses below middle band AND volume falls below average)
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_bb_middle = np.roll(indicators['bollinger']['middle'], 1)
        prev_bb_middle[0] = np.nan

        exit_long_cross_up = (close > indicators['bollinger']['middle']) & (prev_close <= prev_bb_middle)

        # Exit short: close crosses below middle band OR (close crosses above middle band AND volume falls below average)
        exit_short_cross_down = (close < indicators['bollinger']['middle']) & (prev_close >= prev_bb_middle)

        # Volatility gating
        volatility_filter_period = params["volatility_filter_period"]
        atr_avg = np.convolve(atr, np.ones(volatility_filter_period)/volatility_filter_period, mode='valid')
        atr_avg = np.pad(atr_avg, (volatility_filter_period - 1, 0), constant_values=np.nan)
        volatility_gate = atr > atr_avg

        # Apply signals
        long_mask = long_entry_cond & volatility_gate
        short_mask = short_entry_cond & volatility_gate

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Write SL/TP levels
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        return signals