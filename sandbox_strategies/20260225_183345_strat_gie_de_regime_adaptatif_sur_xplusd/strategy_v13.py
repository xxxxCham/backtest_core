from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='xplusdc_regime_adaptive_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_threshold_mult': 1.5,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold_mult': ParameterSpec(
                name='atr_threshold_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.4,
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

        # Prepare indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        obv = np.nan_to_num(indicators['obv'])

        # Previous values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # Volatility regime
        atr_threshold = params.get("atr_threshold_mult", 1.5) * np.mean(atr)
        high_vol = atr > atr_threshold

        # Long entry conditions
        long_cond_high = (
            high_vol
            & (close > prev_close + atr)
            & (obv > prev_obv)
        )
        long_cond_low = (
            ~high_vol
            & (close < prev_close - atr)
            & (adx_val < 25)
            & (obv > prev_obv)
        )
        long_mask = long_cond_high | long_cond_low

        # Short entry conditions
        short_cond_high = (
            high_vol
            & (close < prev_close - atr)
            & (obv < prev_obv)
        )
        short_cond_low = (
            ~high_vol
            & (close > prev_close + atr)
            & (adx_val < 25)
            & (obv < prev_obv)
        )
        short_mask = short_cond_high | short_cond_low

        # Apply entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        # cross_down for long exit
        cross_down_long = (
            (close < prev_close - atr)
            & (prev_close >= prev_close - atr)
        )
        # cross_up for short exit
        cross_up_short = (
            (close > prev_close + atr)
            & (prev_close <= prev_close + atr)
        )
        exit_long = cross_down_long
        exit_short = cross_up_short
        # regime exit
        regime_exit = high_vol & (adx_val > 25)
        exit_mask = exit_long | exit_short | regime_exit
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.4)
        tp_atr_mult = params.get("tp_atr_mult", 2.4)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
