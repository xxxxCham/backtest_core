from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aroon_atr_countertrend')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'atr', 'volume_oscillator', 'sma', 'standard_deviation']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 20,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=21,
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
                default=3.0,
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

        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        sma = np.nan_to_num(indicators['sma'])
        atr = np.nan_to_num(indicators['atr'])

        prev_aroon_up = np.roll(indicators['aroon']['aroon_up'], 1)
        prev_aroon_down = np.roll(indicators['aroon']['aroon_down'], 1)
        prev_aroon_up[0] = np.nan
        prev_aroon_down[0] = np.nan

        cross_up = (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (prev_aroon_down <= prev_aroon_up)
        cross_down = (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up']) & (prev_aroon_down >= prev_aroon_up)

        long_condition = cross_up & (indicators['aroon']['aroon_down'] < 0) & (volume_oscillator > 0) & (df["close"] > sma)
        short_condition = cross_down & (indicators['aroon']['aroon_down'] > 0) & (volume_oscillator < 0) & (df["close"] < sma)

        long_mask = long_condition
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        signals.iloc[:warmup] = 0.0

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        entry_price_long = df["close"] * signals[signals == 1.0]
        entry_price_short = df["close"] * signals[signals == -1.0]

        df.loc[signals == 1.0, "bb_stop_long"] = entry_price_long - stop_atr_mult * atr
        df.loc[signals == 1.0, "bb_tp_long"] = entry_price_long + tp_atr_mult * atr

        df.loc[signals == -1.0, "bb_stop_short"] = entry_price_short + stop_atr_mult * atr
        df.loc[signals == -1.0, "bb_tp_short"] = entry_price_short - tp_atr_mult * atr
        signals.iloc[:warmup] = 0.0
        return signals
