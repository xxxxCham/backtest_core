from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='frontusdc_mean_reversion_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
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
        # Extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])
        # EMA bands
        ema_lower = ema
        ema_upper = ema
        # Volume EMA
        volume_ema = np.nan_to_num(np.convolve(volume_oscillator, np.ones(params["volume_oscillator_fast"])/params["volume_oscillator_fast"], mode='valid'))
        volume_ema = np.pad(volume_ema, (params["volume_oscillator_fast"] - 1, 0), 'constant', constant_values=np.nan)
        # Entry conditions
        # Long entry: close touches lower EMA, volume oscillator above EMA of volume oscillator, Williams R < -80
        long_entry_condition = (close == ema_lower) & (volume_oscillator > volume_ema) & (williams_r < -80)
        long_mask = long_entry_condition
        # Short entry: close touches upper EMA, volume oscillator above EMA of volume oscillator, Williams R > -20
        short_entry_condition = (close == ema_upper) & (volume_oscillator > volume_ema) & (williams_r > -20)
        short_mask = short_entry_condition
        # Exit conditions
        ema_50 = np.nan_to_num(indicators['ema'])
        # Exit long: close crosses above EMA middle band OR williams_r > -20
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_50[0] = np.nan
        exit_long_condition = (close > ema_50) | (williams_r > -20)
        exit_long_mask = exit_long_condition
        # Exit short: close crosses below EMA middle band OR williams_r < -80
        exit_short_condition = (close < ema_50) | (williams_r < -80)
        exit_short_mask = exit_short_condition
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Apply exit conditions to existing positions
        # For simplicity, we assume only one position at a time
        # Find where existing long positions are closed
        long_position = (np.roll(signals, 1) == 1.0) & (signals == 0.0)
        signals[long_position & exit_long_mask] = 0.0
        # Find where existing short positions are closed
        short_position = (np.roll(signals, 1) == -1.0) & (signals == 0.0)
        signals[short_position & exit_short_mask] = 0.0
        # Write SL/TP columns into df if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
