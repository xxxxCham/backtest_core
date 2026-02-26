from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_aroon_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
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

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Indicator access
        bollinger = indicators['bollinger']
        aroon = indicators['aroon']
        atr = indicators['atr']

        # Extract indicator values
        upper = np.nan_to_num(indicators['bollinger']["upper"])
        lower = np.nan_to_num(indicators['bollinger']["lower"])
        aroon_down = np.nan_to_num(indicators['aroon']["aroon_down"])
        aroon_up = np.nan_to_num(indicators['aroon']["aroon_up"])
        atr_val = np.nan_to_num(atr)

        # Get hour from datetime index
        hour = df.index.hour.values

        # Long condition
        long_condition = (
            (df["close"] < lower)
            & (aroon_down < 25)
            & (hour < 6)
        )
        long_mask = long_condition

        # Short condition
        short_condition = (
            (df["close"] > upper)
            & (aroon_up > 75)
            & (hour < 6)
        )
        short_mask = short_condition

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit condition
        exit_condition_long = (df["close"] > upper) | (aroon_up > aroon_down)
        exit_condition_short = (df["close"] < lower) | (aroon_up > aroon_down)

        signals[exit_condition_long & (signals == 1.0)] = 0.0
        signals[exit_condition_short & (signals == -1.0)] = 0.0

        # ATR-based SL/TP (example)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        entry_mask_long = (signals == 1.0)
        df.loc[entry_mask_long, "bb_stop_long"] = df.loc[entry_mask_long, "close"] - stop_atr_mult * atr_val[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = df.loc[entry_mask_long, "close"] + tp_atr_mult * atr_val[entry_mask_long]

        entry_mask_short = (signals == -1.0)
        df.loc[entry_mask_short, "bb_stop_short"] = df.loc[entry_mask_short, "close"] + stop_atr_mult * atr_val[entry_mask_short]
        df.loc[entry_mask_short, "bb_tp_short"] = df.loc[entry_mask_short, "close"] - tp_atr_mult * atr_val[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals