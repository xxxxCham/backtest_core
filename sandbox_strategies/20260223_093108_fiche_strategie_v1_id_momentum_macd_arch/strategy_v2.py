from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_adx_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'rsi_overbought': 75,
            'rsi_oversold': 35,
            'rsi_period': 21,
            'stop_atr_mult': 1.75,
            'tp_atr_mult': 4.5,
            'warmup': 50,
            # defaults for ADX thresholds used in the logic
            'adx_entry_threshold': 25,
            'adx_exit_threshold': 20,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=21,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
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
                default=4.5,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Extract indicator arrays and ensure no NaNs
        macd_vals = np.nan_to_num(indicators['macd']["macd"])
        signal_vals = np.nan_to_num(indicators['macd']["signal"])
        histogram = np.nan_to_num(indicators['macd']["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Cross detection for MACD
        prev_macd = np.roll(macd_vals, 1)
        prev_sig = np.roll(signal_vals, 1)
        prev_macd[0] = np.nan
        prev_sig[0] = np.nan
        cross_up = (macd_vals > signal_vals) & (prev_macd <= prev_sig)
        cross_down = (macd_vals < signal_vals) & (prev_macd >= prev_sig)

        # Sign change detection for histogram
        prev_hist = np.roll(histogram, 1)
        prev_hist[0] = np.nan
        sign_change = ((histogram > 0) & (prev_hist <= 0)) | ((histogram < 0) & (prev_hist >= 0))

        # Thresholds from params or defaults
        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 75)
        adx_entry_threshold = params.get('adx_entry_threshold', 25)
        adx_exit_threshold = params.get('adx_exit_threshold', 20)

        # Long and short entry masks
        long_mask = (
            cross_up
            & (rsi > rsi_oversold)
            & (rsi < rsi_overbought)
            & (adx_val > adx_entry_threshold)
        )
        short_mask = (
            cross_down
            & (rsi > rsi_oversold)
            & (rsi < rsi_overbought)
            & (adx_val > adx_entry_threshold)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit mask
        exit_mask = (
            sign_change
            | (rsi > 80)
            | (rsi < 20)
            | (adx_val < adx_exit_threshold)
        )
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]

        return signals