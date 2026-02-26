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
         'stochastic_overbought': 80,
         'stochastic_oversold': 20,
         'stochastic_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
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

        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Previous values for crossovers
        prev_ema = np.roll(ema, 1)
        prev_obv = np.roll(obv, 1)
        prev_volume_oscillator = np.roll(volume_oscillator, 1)
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        # Entry conditions
        # Long entry: close crosses above ema, OBV rising, volume oscillator increasing and positive
        close_above_ema = (close > ema)
        obv_rising = (obv > prev_obv)
        volume_rising = (volume_oscillator > prev_volume_oscillator)
        volume_positive = (volume_oscillator > 0)

        long_entry = close_above_ema & obv_rising & volume_rising & volume_positive

        # Short entry: close crosses below ema, OBV falling, volume oscillator decreasing and negative
        close_below_ema = (close < ema)
        obv_falling = (obv < prev_obv)
        volume_falling = (volume_oscillator < prev_volume_oscillator)
        volume_negative = (volume_oscillator < 0)

        short_entry = close_below_ema & obv_falling & volume_falling & volume_negative

        # Exit conditions
        # Exit long: close crosses below ema OR volume oscillator crosses below zero
        close_below_ema_exit = (close < ema)
        volume_cross_below_zero = (volume_oscillator < 0) & (prev_volume_oscillator >= 0)
        long_exit = close_below_ema_exit | volume_cross_below_zero

        # Exit short: close crosses above ema OR volume oscillator crosses above zero
        close_above_ema_exit = (close > ema)
        volume_cross_above_zero = (volume_oscillator > 0) & (prev_volume_oscillator <= 0)
        short_exit = close_above_ema_exit | volume_cross_above_zero

        # Apply entries and exits
        long_mask = long_entry
        short_mask = short_entry

        # Apply exits
        exit_long_mask = long_exit & (np.roll(signals, 1) == 1.0)
        exit_short_mask = short_exit & (np.roll(signals, 1) == -1.0)

        # Combine all masks
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Risk management: ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entries
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # Short entries
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
