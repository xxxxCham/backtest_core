from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_rsi_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 1.25,
            'tp_atr_mult': 5.5,
            'warmup': 50,
            'adx_threshold': 25,          # added to avoid KeyError
        }

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
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=1,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Extract indicator arrays safely
        close = df["close"].values
        rsi = np.nan_to_num(indicators['rsi'])
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Long and short entry conditions
        long_mask = (
            (close < lower) &
            (rsi < params["rsi_oversold"]) &
            (adx < params.get("adx_threshold", 25))
        )

        short_mask = (
            (close > upper) &
            (rsi > params["rsi_overbought"]) &
            (adx < params.get("adx_threshold", 25))
        )

        # Exit conditions via cross any
        prev_close = np.roll(close, 1)
        prev_mid = np.roll(middle, 1)
        prev_close[0] = np.nan
        prev_mid[0] = np.nan
        cross_close_mid = (
            (close > middle) & (prev_close <= prev_mid)
        ) | ((close < middle) & (prev_close >= prev_mid))

        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan
        cross_rsi_50 = (
            (rsi > 50.0) & (prev_rsi <= 50.0)
        ) | ((rsi < 50.0) & (prev_rsi >= 50.0))

        exit_mask = cross_close_mid | cross_rsi_50

        # Apply warmup
        signals.iloc[:warmup] = 0.0

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        df.loc[long_mask, "bb_stop_long"] = (
            close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        )
        df.loc[long_mask, "bb_tp_long"] = (
            close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        )

        # ATR-based SL/TP for short entries
        df.loc[short_mask, "bb_stop_short"] = (
            close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        )
        df.loc[short_mask, "bb_tp_short"] = (
            close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        )

        signals.iloc[:warmup] = 0.0
        return signals