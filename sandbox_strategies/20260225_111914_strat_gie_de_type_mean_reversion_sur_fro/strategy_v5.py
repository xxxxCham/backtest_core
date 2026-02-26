from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_frontusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50,
         'williams_r_overbought': -20,
         'williams_r_oversold': -80,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])
        # entry conditions
        close = df["close"].values
        volume_ema = np.nan_to_num(np.roll(volume_oscillator, 1))
        volume_ema[0] = 0.0
        long_condition = (close <= ema_fast) & (volume_oscillator > volume_ema) & (williams_r < params["williams_r_oversold"])
        short_condition = (close >= ema_fast) & (volume_oscillator > volume_ema) & (williams_r > params["williams_r_oversold"])
        long_mask[long_condition] = True
        short_mask[short_condition] = True
        # exit conditions
        ema_slow = np.nan_to_num(indicators['ema'])
        exit_long = (close > ema_slow) | (williams_r > params["williams_r_overbought"])
        exit_short = (close < ema_slow) | (williams_r < params["williams_r_overbought"])
        # apply exits
        signals[long_mask & exit_long] = 0.0
        signals[short_mask & exit_short] = 0.0
        # set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
