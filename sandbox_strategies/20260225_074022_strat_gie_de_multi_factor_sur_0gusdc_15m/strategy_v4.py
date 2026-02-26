from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rsi_adx_bollinger_atr_multi_factor')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'adx', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1,
                max_val=3,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.6,
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
        # Extract price series
        close = df["close"].values

        # Extract indicators with NaN handling
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])

        rsi = np.nan_to_num(indicators['rsi'])

        adx_dict = indicators['adx']
        adx_val = np.nan_to_num(adx_dict["adx"])

        atr = np.nan_to_num(indicators['atr'])

        # Parameter thresholds
        rsi_long_thr = params.get("rsi_long_thr", 55)
        rsi_short_thr = params.get("rsi_short_thr", 45)
        adx_min = params.get("adx_min", 25)
        adx_exit_thr = params.get("adx_exit_thr", 20)
        stop_atr_mult = params.get("stop_atr_mult", 1.6)
        tp_atr_mult = params.get("tp_atr_mult", 3.6)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        long_mask = (close > indicators['bollinger']['upper']) & (rsi > rsi_long_thr) & (adx_val > adx_min)
        short_mask = (close < indicators['bollinger']['lower']) & (rsi < rsi_short_thr) & (adx_val > adx_min)

        # Exit conditions: price crossing Bollinger middle OR ADX falling below exit threshold
        prev_close = np.roll(close, 1)
        prev_mid = np.roll(indicators['bollinger']['middle'], 1)
        prev_close[0] = np.nan
        prev_mid[0] = np.nan
        cross_up = (close > indicators['bollinger']['middle']) & (prev_close <= prev_mid)
        cross_down = (close < indicators['bollinger']['middle']) & (prev_close >= prev_mid)
        cross_any = cross_up | cross_down
        exit_mask = cross_any | (adx_val < adx_exit_thr)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR‑based stop‑loss and take‑profit on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
