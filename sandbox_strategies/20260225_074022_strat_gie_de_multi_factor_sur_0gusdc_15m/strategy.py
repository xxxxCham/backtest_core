from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='multi_factor_rsi_macd_adx_15m')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'rsi_overbought': 55,
         'rsi_oversold': 45,
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
            'rsi_overbought': ParameterSpec(
                name='rsi_overbought',
                min_val=50,
                max_val=80,
                default=55,
                param_type='int',
                step=1,
            ),
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=20,
                max_val=50,
                default=45,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=15,
                default=9,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
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
        close = df["close"].values
        rsi = np.nan_to_num(indicators['rsi'])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        rsi_ob = float(params.get("rsi_overbought", 55))
        rsi_os = float(params.get("rsi_oversold", 45))
        adx_min = float(params.get("adx_min", 25))
        adx_exit = float(params.get("adx_exit", 20))
        stop_mult = float(params.get("stop_atr_mult", 1.6))
        tp_mult = float(params.get("tp_atr_mult", 3.6))

        # entry conditions
        long_cond = (rsi > rsi_ob) & (macd_hist > 0) & (adx_val > adx_min)
        short_cond = (rsi < rsi_os) & (macd_hist < 0) & (adx_val > adx_min)

        long_mask[:] = long_cond
        short_mask[:] = short_cond

        # exit conditions (more than half of factors reverse)
        exit_long = ((rsi < rsi_ob) & (macd_hist < 0)) | (adx_val < adx_exit)
        exit_short = ((rsi > rsi_os) & (macd_hist > 0)) | (adx_val < adx_exit)

        # override entries with exits
        long_mask[exit_long] = False
        short_mask[exit_short] = False

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # write ATR‑based stop‑loss / take‑profit on entry bars
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
