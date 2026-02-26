from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_breakout_ichimoku_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'atr', 'volume_oscillator']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_min': 0.002,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.84,
         'volume_osc_fast': 14,
         'volume_osc_slow': 28,
         'warmup': 60}

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
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.0005,
                max_val=0.01,
                default=0.002,
                param_type='float',
                step=0.1,
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
                max_val=6.0,
                default=3.84,
                param_type='float',
                step=0.1,
            ),
            'volume_osc_fast': ParameterSpec(
                name='volume_osc_fast',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'volume_osc_slow': ParameterSpec(
                name='volume_osc_slow',
                min_val=10,
                max_val=60,
                default=28,
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
        # initialize signals series (already provided by caller)
        n = len(df)

        # extract price series
        close = df["close"].values

        # extract and sanitize indicators
        atr = np.nan_to_num(indicators['atr'])
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])

        # parameters
        atr_min = float(params.get("atr_min", 0.002))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.6))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.84))
        warmup = int(params.get("warmup", 50))

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # entry conditions
        long_mask = (close > senkou_a) & (vol_osc > 0) & (atr > atr_min)
        short_mask = (close < senkou_b) & (vol_osc < 0) & (atr > atr_min)

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # prepare SL/TP columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute SL/TP levels for longs
        if long_mask.any():
            stop_long = close - stop_atr_mult * atr
            tp_long = close + tp_atr_mult * atr
            df.loc[long_mask, "bb_stop_long"] = stop_long[long_mask]
            df.loc[long_mask, "bb_tp_long"] = tp_long[long_mask]

        # compute SL/TP levels for shorts
        if short_mask.any():
            stop_short = close + stop_atr_mult * atr
            tp_short = close - tp_atr_mult * atr
            df.loc[short_mask, "bb_stop_short"] = stop_short[short_mask]
            df.loc[short_mask, "bb_tp_short"] = tp_short[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
