from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_bollinger_volume_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
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
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        bollinger = indicators['bollinger']
        upper = np.nan_to_num(indicators['bollinger']["upper"])
        middle = np.nan_to_num(indicators['bollinger']["middle"])
        lower = np.nan_to_num(indicators['bollinger']["lower"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Compute EMA values for fast and slow
        ema_50 = ema_fast
        ema_200 = ema_slow

        # Compute bandwidth
        bandwidth = (upper - lower) / middle

        # Compute crossovers
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_200 = np.roll(ema_200, 1)
        prev_ema_50[0] = np.nan
        prev_ema_200[0] = np.nan
        cross_up_50_200 = (ema_50 > ema_200) & (prev_ema_50 <= prev_ema_200)
        cross_down_50_200 = (ema_50 < ema_200) & (prev_ema_50 >= prev_ema_200)

        # Entry conditions
        narrow_band = bandwidth < 0.02
        volume_positive = volume_oscillator > 0
        volume_negative = volume_oscillator < 0

        # Long entry
        long_entry = cross_up_50_200 & narrow_band & volume_positive
        long_mask = long_entry

        # Short entry
        short_entry = cross_down_50_200 & narrow_band & volume_negative
        short_mask = short_entry

        # Exit conditions
        rsi_overbought = rsi > 70
        exit_cross_down_200 = (ema_50 < ema_200) & (prev_ema_50 >= prev_ema_200)
        exit_long = rsi_overbought | exit_cross_down_200
        exit_short = rsi_overbought | exit_cross_down_200

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
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