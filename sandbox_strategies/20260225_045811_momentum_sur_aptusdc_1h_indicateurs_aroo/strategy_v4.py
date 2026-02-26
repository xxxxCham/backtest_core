from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_momentum_v4')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'macd', 'onchain_smoothing', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'onchain_smoothing_period': 20,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=10,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=30,
                default=12,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
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
        signals.iloc[:warmup] = 0.0

        aroon = indicators['aroon']
        macd = indicators['macd']
        onchain_smoothing = np.nan_to_num(indicators['onchain_smoothing'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Long entry condition
        aroon_up_cross = np.roll(indicators['aroon']["aroon_up"], 1) < indicators['aroon']["aroon_up"]
        macd_bullish_crossover = (indicators['macd']["macd"] > indicators['macd']["signal"]) & (np.roll(indicators['macd']["macd"], 1) <= np.roll(indicators['macd']["signal"], 1))
        onchain_smoothing_increasing = onchain_smoothing > np.roll(onchain_smoothing, 1)

        long_mask = aroon_up_cross & macd_bullish_crossover & onchain_smoothing_increasing

        # Short entry condition (not implemented as per instructions)
        # short_mask = ...

        # Exit conditions
        aroon_down_cross = np.roll(indicators['aroon']["aroon_down"], 1) > indicators['aroon']["aroon_down"]
        macd_bearish_crossover = (indicators['macd']["macd"] < indicators['macd']["signal"]) & (np.roll(indicators['macd']["macd"], 1) >= np.roll(indicators['macd']["signal"], 1))

        exit_mask = aroon_down_cross | macd_bearish_crossover

        signals[long_mask] = 1.0
        signals[exit_mask] = 0.0  # Exit long positions

        # ATR-based stop loss and take profit
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        entry_mask = long_mask

        df.loc[entry_mask, "bb_stop_long"] = close[entry_mask] - stop_atr_mult * atr[entry_mask]
        df.loc[entry_mask, "bb_tp_long"] = close[entry_mask] + tp_atr_mult * atr[entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
