from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='xplusdc_30m_obv_atr_ema')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'ema', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_threshold': 0.5,
         'ema_period': 20,
         'leverage': 1,
         'obv_sma_period': 20,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 2.4,
         'warmup': 50}

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
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'obv_sma_period': ParameterSpec(
                name='obv_sma_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
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
                max_val=5.0,
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

        signals.iloc[:warmup] = 0.0

        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        obv = np.nan_to_num(indicators['obv'])

        # OBV SMA
        obv_window = int(params.get("obv_sma_period", 20))
        if obv_window > 1:
            obv_sma_valid = np.convolve(obv, np.ones(obv_window) / obv_window, mode="valid")
            obv_sma = np.concatenate([np.full(obv_window - 1, np.nan), obv_sma_valid])
        else:
            obv_sma = obv

        atr_threshold = params.get("atr_threshold", 0.5)

        # Entry conditions
        long_mask = (close > ema) & (obv > obv_sma) & (atr > atr_threshold)
        short_mask = (close < ema) & (obv < obv_sma) & (atr > atr_threshold)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: cross of close and ema or low volatility
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        prev_close[0] = np.nan
        prev_ema[0] = np.nan
        cross_up = (close > ema) & (prev_close <= prev_ema)
        cross_down = (close < ema) & (prev_close >= prev_ema)
        cross_any = cross_up | cross_down
        low_vol = atr < atr_threshold
        exit_mask = cross_any | low_vol
        # Do not overwrite entry signals on same bar
        exit_mask = exit_mask & (signals == 0.0)
        signals[exit_mask] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = params.get("stop_atr_mult", 1.4)
        tp_mult = params.get("tp_atr_mult", 2.4)

        long_entries = signals == 1.0
        short_entries = signals == -1.0

        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - stop_mult * atr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + tp_mult * atr[long_entries]

        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + stop_mult * atr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - tp_mult * atr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals
