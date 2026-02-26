from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='improved_mean_reversion_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
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
                default=3.0,
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
        # Extract and sanitize indicator arrays
        bollinger = indicators['bollinger']
        atr = np.nan_to_num(indicators['atr'])
        rsi_val = np.nan_to_num(indicators['rsi'])

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Entry conditions

        # Long conditions: price rebounds from lower Bollinger AND RSI < oversold
        indicators['bollinger']['lower'] = np.nan_to_num(indicators['bollinger']["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(indicators['bollinger']["middle"])
        prev_lower = np.roll(indicators['bollinger']['lower'], 1)
        prev_lower[0] = np.nan

        long_rebound = (df["close"] > indicators['bollinger']['lower']) & (df["close"] <= indicators['bollinger']['middle']) & \
                       (prev_lower > df["close"]) & (rsi_val < params["rsi_oversold"])

        # Short conditions: price rebounds from upper Bollinger AND RSI > overbought
        indicators['bollinger']['upper'] = np.nan_to_num(indicators['bollinger']["upper"])
        prev_upper = np.roll(indicators['bollinger']['upper'], 1)
        prev_upper[0] = np.nan

        short_rebound = (df["close"] < indicators['bollinger']['upper']) & (df["close"] >= indicators['bollinger']['middle']) & \
                        (prev_upper < df["close"]) & (rsi_val > params["rsi_overbought"])

        # Exit conditions
        exit_middle = (rsi_val < params["rsi_overbought"]) & \
                      (rsi_val > params["rsi_oversold"])
        exit_cross = (rsi_val > params["rsi_oversold"]) & \
                     (rsi_val < params["rsi_overbought"])

        # Apply signals
        long_mask = long_rebound & ~exit_middle
        short_mask = short_rebound & ~exit_cross

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management - SL/TP columns

        close_arr = df["close"].values
        signals_arr = signals.values

        entry_long_mask = (signals_arr == 1.0) & long_rebound
        entry_short_mask = (signals_arr == -1.0) & short_rebound

        # Long SL/TP
        df.loc[:, "sl_level"] = np.nan
        df.loc[:, "tp_level"] = np.nan
        df.loc[entry_long_mask, "sl_level"] = close_arr[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "tp_level"] = close_arr[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        signals.iloc[:warmup] = 0.0
        return signals
