from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd_with_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'rsi', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_period': 18,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=18,
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
                min_val=1.5,
                max_val=5.0,
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

        # unwrap indicators
        indicators['macd']['macd'] = np.nan_to_num(indicators['macd']["macd"])
        signal_line = np.nan_to_num(indicators['macd']["signal"])
        hist = np.nan_to_num(indicators['macd']["histogram"])
        rsi = np.nan_to_num(indicators['rsi'])
        adx = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # cross up/down for MACD
        prev_macd = np.roll(indicators['macd']['macd'], 1)
        prev_signal = np.roll(signal_line, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        cross_up_macd = (indicators['macd']['macd'] > signal_line) & (prev_macd <= prev_signal)
        cross_down_macd = (indicators['macd']['macd'] < signal_line) & (prev_macd >= prev_signal)

        # sign change of histogram
        prev_hist = np.roll(hist, 1)
        prev_hist[0] = np.nan
        sign_change_hist = (hist > 0) != (prev_hist > 0)

        # exit mask
        exit_mask = (
            sign_change_hist
            | (rsi > 80)
            | (rsi < 20)
            | (adx < 20)
        )

        # long / short entry masks
        long_mask = (
            cross_up_macd
            & (rsi > 40)
            & (rsi < 65)
            & (adx > 25)
            & ~exit_mask
        )
        short_mask = (
            cross_down_macd
            & (rsi > 30)
            & (rsi < 60)
            & (adx > 25)
            & ~exit_mask
        )

        signals[exit_mask] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        if np.any(signals == 1.0):
            long_entry = signals == 1.0
            df.loc[long_entry, "bb_stop_long"] = close[long_entry] - params["stop_atr_mult"] * atr[long_entry]
            df.loc[long_entry, "bb_tp_long"] = close[long_entry] + params["tp_atr_mult"] * atr[long_entry]
        if np.any(signals == -1.0):
            short_entry = signals == -1.0
            df.loc[short_entry, "bb_stop_short"] = close[short_entry] + params["stop_atr_mult"] * atr[short_entry]
            df.loc[short_entry, "bb_tp_short"] = close[short_entry] - params["tp_atr_mult"] * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
