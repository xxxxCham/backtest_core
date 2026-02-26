from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_aroon_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=15,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
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
                default=2.5,
                param_type='float',
                step=0.1,
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

        kelt = indicators['keltner']
        aroon = indicators['aroon']
        atr = indicators['atr']

        upper = np.nan_to_num(kelt["upper"])
        lower = np.nan_to_num(kelt["lower"])
        middle = np.nan_to_num(kelt["middle"])
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        atr_val = np.nan_to_num(atr)

        # Long entry
        long_condition = (df["close"] < lower) & (indicators['aroon']['aroon_down'] > 70)
        long_mask = long_condition

        # Short entry
        short_condition = (df["close"] > upper) & (indicators['aroon']['aroon_up'] > 70)
        short_mask = short_condition

        # Exit condition: close crosses keltner middle band
        exit_condition = (df["close"] > middle) | (df["close"] < middle)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Reset signals on exit
        signals[exit_condition] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based stop loss and take profit (write to df)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan

        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = df.loc[entry_long_mask, "close"] - stop_atr_mult * atr_val[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = df.loc[entry_long_mask, "close"] + tp_atr_mult * atr_val[entry_long_mask]
        signals.iloc[:warmup] = 0.0
        return signals
