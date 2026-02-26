from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_rsi_atr_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.0,
         'ema_period': 50,
         'leverage': 1,
         'rsi_overbought': 55,
         'rsi_oversold': 45,
         'rsi_period': 14,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 4.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=5.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=10.0,
                default=4.5,
                param_type='float',
                step=0.1,
            ),
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0,
                max_val=10.0,
                default=0.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
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
        # Extract needed series as numpy arrays, ensuring NaNs are handled
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        rsi_ob = float(params.get("rsi_overbought", 55))
        rsi_os = float(params.get("rsi_oversold", 45))
        atr_thr = float(params.get("atr_threshold", 0.0))
        stop_mult = float(params.get("stop_atr_mult", 2.2))
        tp_mult = float(params.get("tp_atr_mult", 4.5))

        # Entry conditions
        long_cond = (close > ema) & (rsi > rsi_ob) & (atr > atr_thr)
        short_cond = (close < ema) & (rsi < rsi_os) & (atr > atr_thr)

        # Prevent simultaneous long and short signals
        both = long_cond & short_cond
        long_cond = long_cond & ~both
        short_cond = short_cond & ~both

        # Apply to masks
        long_mask[long_cond] = True
        short_mask[short_cond] = True

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Write ATR‑based stop‑loss and take‑profit levels on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        # Warmup protection (already set in skeleton, kept for safety)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
