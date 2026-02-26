from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 10,
         'atr_period': 14,
         'donchian_period': 15,
         'leverage': 1,
         'stop_atr_mult': 1.25,
         'tp_atr_mult': 5.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=1,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=1,
                max_val=50,
                default=15,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=1,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.5,
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
        donchian = indicators['donchian']
        adx = np.nan_to_num(indicators['adx']["adx"])
        close = df["close"].values
        prev_highs = np.roll(df['high'], 1)
        prev_lows = np.roll(df['low'], 1)

        # Entry long
        long_mask = (close > indicators['donchian']["upper"]) & (adx > 30)
        signals[long_mask] = 1.0

        # Exit long
        exit_mask = ((close < indicators['donchian']["middle"]) | (prev_highs >= df['high']) | (prev_lows <= df['low'])) & (signals == 1.0)
        signals[exit_mask] = 0.0

        # Entry short
        short_mask = (close < indicators['donchian']["lower"]) & (adx > 30)
        signals[short_mask] = -1.0

        # Exit short
        exit_mask = ((close > indicators['donchian']["middle"]) | (prev_highs >= df['high']) | (prev_lows <= df['low'])) & (signals == -1.0)
        signals[exit_mask] = 0.0

        # Set stop and take profit levels for long entries
        entry_price = df["close"][long_mask]
        atr = np.nan_to_num(indicators['atr'])
        stop_multiplier = params.get("stop_atr_mult", 1.25)
        tp_multiplier = params.get("tp_atr_mult", 5.5)
        df.loc[long_mask, "bb_stop_long"] = entry_price - stop_multiplier * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = entry_price + tp_multiplier * atr[long_mask]
        signals.iloc[:warmup] = 0.0
        return signals
