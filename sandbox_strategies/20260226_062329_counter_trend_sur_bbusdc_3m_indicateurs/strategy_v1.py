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
         'sma_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'trade_session_hours': 2,
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

        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        atr = np.nan_to_num(indicators['atr'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        sma = np.nan_to_num(indicators['sma'])

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        aroon_cross_up = (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (np.roll(indicators['aroon']['aroon_down'], 1) <= np.roll(indicators['aroon']['aroon_up'], 1))
        aroon_down_negative = indicators['aroon']['aroon_down'] < 0
        volume_oscillator_positive = volume_oscillator > 0
        close_above_sma = df["close"] > sma

        aroon_cross_down = (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up']) & (np.roll(indicators['aroon']['aroon_down'], 1) >= np.roll(indicators['aroon']['aroon_up'], 1))
        aroon_down_positive = indicators['aroon']['aroon_down'] > 0
        volume_oscillator_negative = volume_oscillator < 0
        close_below_sma = df["close"] < sma

        # Long entry
        long_mask = aroon_cross_up & aroon_down_negative & volume_oscillator_positive & close_above_sma

        # Short entry
        short_mask = aroon_cross_down & aroon_down_positive & volume_oscillator_negative & close_below_sma

        # Exit conditions
        aroon_down_cross_positive = indicators['aroon']['aroon_down'] > 0
        volume_oscillator_negative_exit = volume_oscillator < 0

        # Exit logic
        exit_mask = aroon_down_cross_positive | volume_oscillator_negative_exit

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based SL/TP (write to DataFrame)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        entry_mask_long = (signals == 1.0)
        entry_mask_short = (signals == -1.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[entry_mask_long, "bb_stop_long"] = df.loc[entry_mask_long, "close"] - stop_atr_mult * atr[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = df.loc[entry_mask_long, "close"] + tp_atr_mult * atr[entry_mask_long]

        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # df.loc[entry_mask_short, "bb_stop_short"] = df.loc[entry_mask_short, "close"] + stop_atr_mult * atr[entry_mask_short]
        # df.loc[entry_mask_short, "bb_tp_short"] = df.loc[entry_mask_short, "close"] - tp_atr_mult * atr[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals
