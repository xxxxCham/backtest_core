from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bb_aroon_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'atr', 'volume_oscillator', 'sma', 'standard_deviation']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 20,
         'atr_period': 14,
         'leverage': 1,
         'sma_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'trading_hours': 120,
         'volume_oscillator_period': 14,
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
                min_val=10,
                max_val=20,
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

        aroon = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        atr = np.nan_to_num(indicators['atr'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        sma = np.nan_to_num(indicators['sma'])
        standard_deviation = np.nan_to_num(indicators['standard_deviation'])

        # Entry conditions
        long_condition = (np.roll(indicators['aroon']['aroon_down'], 1) > indicators['aroon']['aroon_up']) & (indicators['aroon']['aroon_down'] < 0) & (volume_oscillator > 0) & (df['close'] > sma)
        short_condition = (np.roll(indicators['aroon']['aroon_down'], 1) < indicators['aroon']['aroon_up']) & (indicators['aroon']['aroon_down'] > 0) & (volume_oscillator < 0) & (df['close'] < sma)

        long_mask = long_condition
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        aroon_down_cross_zero = (indicators['aroon']['aroon_down'] < 0) & (np.roll(indicators['aroon']['aroon_down'], 1) >= 0)
        volume_oscillator_negative = volume_oscillator < 0

        exit_long_mask = aroon_down_cross_zero | volume_oscillator_negative
        exit_short_mask = aroon_down_cross_zero | volume_oscillator_negative

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # ATR-based stop-loss and take-profit (write to DataFrame)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[entry_long_mask, "bb_stop_long"] = df.loc[entry_long_mask, "close"] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = df.loc[entry_long_mask, "close"] + tp_atr_mult * atr[entry_long_mask]

        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        df.loc[entry_short_mask, "bb_stop_short"] = df.loc[entry_short_mask, "close"] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = df.loc[entry_short_mask, "close"] - tp_atr_mult * atr[entry_short_mask]

        # Warmup protection
        signals.iloc[:warmup] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
