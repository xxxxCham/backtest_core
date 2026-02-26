from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_aroon_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'atr_period': 14,
         'bollinger_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=5.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=10,
                max_val=50,
                default=25,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Indicators
        bollinger = indicators['bollinger']
        aroon = indicators['aroon']
        atr = np.nan_to_num(indicators['atr'])

        upper = np.nan_to_num(indicators['bollinger']["upper"])
        lower = np.nan_to_num(indicators['bollinger']["lower"])
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])

        # Entry conditions
        long_condition = (df["close"] < lower) & (indicators['aroon']['aroon_down'] < 20) & (df.index.hour < 6)
        short_condition = (df["close"] > upper) & (indicators['aroon']['aroon_up'] > 80) & (df.index.hour < 6)

        # Assign signals
        long_mask = long_condition
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        cross_middle = (df["close"] > np.roll(upper, 1)) | (df["close"] < np.roll(lower, 1))
        aroon_cross = (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) | (indicators['aroon']['aroon_up'] < indicators['aroon']['aroon_down'])

        exit_mask = cross_middle | aroon_cross
        signals[exit_mask] = 0.0

        # ATR-based SL/TP (write to df columns)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = df.loc[entry_long_mask, "close"] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = df.loc[entry_long_mask, "close"] + tp_atr_mult * atr[entry_long_mask]

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
