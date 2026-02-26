from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='volume_ema_momentum')

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
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=100,
                max_val=300,
                default=200,
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
        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # Prepare EMA arrays
        ema_50 = ema_fast
        ema_200 = ema_slow
        # Prepare OBV and Volume Oscillator
        obv_ma = np.convolve(obv, np.ones(params["obv_period"])/params["obv_period"], mode='valid')
        obv_ma = np.pad(obv_ma, (params["obv_period"] - 1, 0), mode='constant', constant_values=np.nan)
        volume_oscillator_ma = np.convolve(volume_oscillator, np.ones(params["volume_oscillator_fast"])/params["volume_oscillator_fast"], mode='valid')
        volume_oscillator_ma = np.pad(volume_oscillator_ma, (params["volume_oscillator_fast"] - 1, 0), mode='constant', constant_values=np.nan)
        # Compute previous values for crossovers
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_200 = np.roll(ema_200, 1)
        prev_ema_50[0] = np.nan
        prev_ema_200[0] = np.nan
        # Compute crossover masks
        ema_cross_up = (ema_50 > ema_200) & (prev_ema_50 <= prev_ema_200)
        ema_cross_down = (ema_50 < ema_200) & (prev_ema_50 >= prev_ema_200)
        # Compute volume conditions
        obv_up = obv > obv_ma
        volume_up = volume_oscillator > volume_oscillator_ma
        # Get previous volume_oscillator for trend
        prev_volume_oscillator = np.roll(volume_oscillator, 1)
        prev_volume_oscillator[0] = np.nan
        volume_trend_up = volume_oscillator > prev_volume_oscillator
        # Long entry: EMA crossover up + OBV rising + Volume oscillator rising
        long_condition = ema_cross_up & obv_up & volume_trend_up
        long_mask = long_condition
        # Short entry: EMA crossover down + OBV falling + Volume oscillator falling
        short_condition = ema_cross_down & ~obv_up & ~volume_trend_up
        short_mask = short_condition
        # Exit: EMA crossover down or Volume oscillator falling
        exit_condition = ema_cross_down | (volume_oscillator < prev_volume_oscillator)
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Apply exit signals
        exit_mask = exit_condition
        signals[exit_mask] = 0.0
        # Write SL/TP columns for ATR-based risk management
        close = df["close"].values
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        if np.any(entry_long_mask):
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        if np.any(entry_short_mask):
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
