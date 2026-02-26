from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'obv_period': 10,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 5,
         'volume_oscillator_slow': 10,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=10,
                max_val=100,
                default=50,
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
        ema_fast = np.nan_to_num(indicators['ema'])  # Default EMA length is 20, we'll use it as fast
        ema_slow = np.nan_to_num(indicators['ema'])  # Default EMA length is 50, we'll use it as slow
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        keltner = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(indicators['keltner']["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])
        close = df["close"].values

        # EMA fast and slow
        ema_fast = np.nan_to_num(indicators['ema'])  # Assuming default EMA length = 20
        ema_slow = np.nan_to_num(indicators['ema'])  # Assuming default EMA length = 50

        # Adjusting EMA arrays to correct periods
        # We'll compute the actual EMA arrays based on params
        # But since only one EMA array is provided, we assume it's a generic EMA
        # Let's compute EMA arrays manually for periods specified
        ema_fast_len = params.get("ema_fast", 20)
        ema_slow_len = params.get("ema_slow", 50)
        obv_period = params.get("obv_period", 10)
        volume_oscillator_fast = params.get("volume_oscillator_fast", 5)
        volume_oscillator_slow = params.get("volume_oscillator_slow", 10)

        # Compute rolling means for OBV and volume oscillator
        obv_mean = np.full(n, np.nan)
        volume_oscillator_mean = np.full(n, np.nan)

        # Compute rolling means manually using numpy
        obv_mean[obv_period - 1:] = np.convolve(obv, np.ones(obv_period), 'valid') / obv_period
        volume_oscillator_mean[volume_oscillator_slow - 1:] = np.convolve(volume_oscillator, np.ones(volume_oscillator_slow), 'valid') / volume_oscillator_slow

        # Pad the beginning with NaN to match length
        obv_mean[:obv_period - 1] = np.nan
        volume_oscillator_mean[:volume_oscillator_slow - 1] = np.nan

        # Crossings
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan

        cross_up = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        cross_down = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)

        # Entry conditions
        # Long entry: EMA crossover up + OBV rising + volume oscillator above mean
        obv_condition_long = obv > obv_mean
        volume_condition_long = volume_oscillator > volume_oscillator_mean

        long_entry = cross_up & obv_condition_long & volume_condition_long

        # Short entry: EMA crossover down + OBV falling + volume oscillator below mean
        obv_condition_short = obv < obv_mean
        volume_condition_short = volume_oscillator < volume_oscillator_mean

        short_entry = cross_down & obv_condition_short & volume_condition_short

        # Exit conditions
        exit_long = cross_down
        exit_short = cross_up

        # Regime filter: only trade if price is outside Keltner Channel
        regime_long = close > indicators['keltner']['upper']
        regime_short = close < indicators['keltner']['lower']

        # Apply filters
        long_mask = long_entry & regime_long
        short_mask = short_entry & regime_short

        # Exit signals
        signals[exit_long] = -1.0
        signals[exit_short] = 1.0

        # Entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

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
