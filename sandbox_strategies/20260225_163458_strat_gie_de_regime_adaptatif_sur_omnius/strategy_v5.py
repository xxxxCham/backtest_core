from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vwap_rsi_adx_atr_regime')

    @property
    def required_indicators(self) -> List[str]:
        return ['vwap', 'rsi', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.5,
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
                max_val=3.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=2.5,
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

        # Wrap indicators
        close = df["close"].values
        vwap = np.nan_to_num(indicators['vwap'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # Parameters
        rsi_ovb = params.get("rsi_overbought", 60.0)
        rsi_ovs = params.get("rsi_oversold", 40.0)
        adx_entry_thr = params.get("adx_entry", 25.0)
        adx_exit_thr = params.get("adx_exit", 20.0)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        # Entry conditions
        long_mask = (close > vwap) & (rsi > rsi_ovb) & (adx_val > adx_entry_thr)
        short_mask = (close < vwap) & (rsi < rsi_ovs) & (adx_val > adx_entry_thr)

        # Exit conditions
        prev_close = np.roll(close, 1); prev_close[0] = np.nan
        prev_vwap = np.roll(vwap, 1); prev_vwap[0] = np.nan
        cross_close_vwap = ((close > vwap) & (prev_close <= prev_vwap)) | \
                           ((close < vwap) & (prev_close >= prev_vwap))

        prev_rsi = np.roll(rsi, 1); prev_rsi[0] = np.nan
        cross_rsi_50 = ((rsi > 50.0) & (prev_rsi <= 50.0)) | \
                       ((rsi < 50.0) & (prev_rsi >= 50.0))

        exit_mask = cross_close_vwap | cross_rsi_50 | (adx_val < adx_exit_thr)

        # Apply warmup
        signals.iloc[:warmup] = 0.0

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long SL/TP
        long_entry = (signals == 1.0)
        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        # Short SL/TP
        short_entry = (signals == -1.0)
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
