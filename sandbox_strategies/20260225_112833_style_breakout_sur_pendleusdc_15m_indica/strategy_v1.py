from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='style_breakout_pendleusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
         'leverage': 1,
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

        ema_period = params.get("ema_period", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Compute rolling means for confirmation
        obv_rolling_mean = np.convolve(obv, np.ones(20)/20, mode='valid')
        obv_rolling_mean = np.pad(obv_rolling_mean, (19, 0), mode='constant')

        volume_oscillator_rolling_mean = np.convolve(volume_oscillator, np.ones(20)/20, mode='valid')
        volume_oscillator_rolling_mean = np.pad(volume_oscillator_rolling_mean, (19, 0), mode='constant')

        # Entry conditions
        # Long entry: close crosses above EMA, OBV increasing, volume oscillator increasing
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        prev_obv = np.roll(obv, 1)
        prev_volume_oscillator = np.roll(volume_oscillator, 1)

        prev_close[0] = np.nan
        prev_ema[0] = np.nan
        prev_obv[0] = np.nan
        prev_volume_oscillator[0] = np.nan

        cross_above_ema = (close > ema) & (prev_close <= prev_ema)

        obv_increasing = (obv > obv_rolling_mean)
        volume_oscillator_increasing = (volume_oscillator > volume_oscillator_rolling_mean)

        long_entry_condition = cross_above_ema & obv_increasing & volume_oscillator_increasing

        # Short entry: close crosses below EMA, OBV decreasing, volume oscillator decreasing
        cross_below_ema = (close < ema) & (prev_close >= prev_ema)

        obv_decreasing = (obv < obv_rolling_mean)
        volume_oscillator_decreasing = (volume_oscillator < volume_oscillator_rolling_mean)

        short_entry_condition = cross_below_ema & obv_decreasing & volume_oscillator_decreasing

        # Exit conditions
        # Exit long: close crosses below EMA with bearish OBV divergence
        exit_long_condition = cross_below_ema & obv_decreasing & volume_oscillator_decreasing

        # Exit short: close crosses above EMA with bullish OBV divergence
        exit_short_condition = cross_above_ema & obv_increasing & volume_oscillator_increasing

        # Set masks
        long_mask = long_entry_condition
        short_mask = short_entry_condition

        # Apply exit conditions
        exit_long_mask = exit_long_condition
        exit_short_mask = exit_short_condition

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit signals
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
