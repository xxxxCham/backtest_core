from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_sur_0gusdc_1h')

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
         'obv_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 10,
         'volume_oscillator_slow': 30,
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
            'volume_oscillator_fast': ParameterSpec(
                name='volume_oscillator_fast',
                min_val=5,
                max_val=30,
                default=10,
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
        keltner = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(indicators['keltner']["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])
        # Compute EMA values
        ema_fast_val = ema_fast
        ema_slow_val = ema_slow
        # Compute rolling means for OBV and volume oscillator
        obv_rolling = np.convolve(obv, np.ones(params["obv_period"])/params["obv_period"], mode='valid')
        obv_rolling = np.pad(obv_rolling, (params["obv_period"] - 1, 0), 'constant', constant_values=np.nan)
        volume_oscillator_rolling = np.convolve(volume_oscillator, np.ones(params["volume_oscillator_fast"])/params["volume_oscillator_fast"], mode='valid')
        volume_oscillator_rolling = np.pad(volume_oscillator_rolling, (params["volume_oscillator_fast"] - 1, 0), 'constant', constant_values=np.nan)
        # Cross detection
        prev_ema_fast = np.roll(ema_fast_val, 1)
        prev_ema_slow = np.roll(ema_slow_val, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        ema_cross_up = (ema_fast_val > ema_slow_val) & (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast_val < ema_slow_val) & (prev_ema_fast >= prev_ema_slow)
        # Confirm entries with OBV and volume oscillator
        obv_confirm_long = obv > obv_rolling
        obv_confirm_short = obv < obv_rolling
        volume_confirm_long = volume_oscillator > volume_oscillator_rolling
        volume_confirm_short = volume_oscillator < volume_oscillator_rolling
        # Long entry conditions
        entry_long = ema_cross_up & obv_confirm_long & volume_confirm_long
        # Short entry conditions
        entry_short = ema_cross_down & obv_confirm_short & volume_confirm_short
        # Exit conditions
        exit_long = ema_cross_down | (volume_oscillator < volume_oscillator_rolling)
        exit_short = ema_cross_up | (volume_oscillator > volume_oscillator_rolling)
        # Regime filter using Keltner Channel
        keltner_volatility = indicators['keltner']['upper'] - indicators['keltner']['lower']
        regime_filter = keltner_volatility > 0
        # Apply signals
        long_mask = entry_long & regime_filter
        short_mask = entry_short & regime_filter
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Exit conditions
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0
        close = df["close"].values
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
