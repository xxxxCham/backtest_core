from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='lctusdc_trend_follow_sma_aroon_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'sma_period': 20,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 6.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
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
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=6.6,
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
        sma = np.nan_to_num(indicators['sma'])

        ar = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(ar["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(ar["aroon_down"])

        atr = np.nan_to_num(indicators['atr'])

        # cross detection
        prev_close = np.roll(close, 1)
        prev_sma = np.roll(sma, 1)
        prev_close[0] = np.nan
        prev_sma[0] = np.nan
        cross_up = (close > sma) & (prev_close <= prev_sma)
        cross_down = (close < sma) & (prev_close >= prev_sma)

        # long entry
        long_entry = cross_up & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) & (indicators['aroon']['aroon_up'] > 50)
        long_mask = long_entry

        # short entry
        short_entry = cross_down & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (indicators['aroon']['aroon_down'] > 50)
        short_mask = short_entry

        # exit conditions
        long_exit = cross_down | ((indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (indicators['aroon']['aroon_down'] > 50))
        short_exit = cross_up | ((indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) & (indicators['aroon']['aroon_up'] > 50))

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        # ATR-based SL/TP for short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
