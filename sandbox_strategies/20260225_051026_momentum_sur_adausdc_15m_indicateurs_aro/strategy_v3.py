from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_momentum_aroon_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'ema', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.5,
         'volume_oscillator_period': 10,
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
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_period': ParameterSpec(
                name='volume_oscillator_period',
                min_val=5,
                max_val=20,
                default=10,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.5,
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

        aroon = indicators['aroon']
        ema = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        ema_direction = np.sign(ema)

        long_condition = (indicators['aroon']["aroon_up"] > indicators['aroon']["aroon_down"]) & (volume_oscillator > 0) & (ema_direction > 0)
        short_condition = (indicators['aroon']["aroon_up"] < indicators['aroon']["aroon_down"]) & (volume_oscillator < 0) & (ema_direction < 0)

        long_mask = long_condition
        short_mask = short_condition

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based stop-loss and take-profit
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        entry_mask_long = signals == 1.0
        entry_prices_long = df["close"].values[entry_mask_long]
        stop_levels_long = entry_prices_long - stop_atr_mult * atr[entry_mask_long]
        tp_levels_long = entry_prices_long + tp_atr_mult * atr[entry_mask_long]

        df.loc[entry_mask_long, "bb_stop_long"] = stop_levels_long
        df.loc[entry_mask_long, "bb_tp_long"] = tp_levels_long
        signals.iloc[:warmup] = 0.0
        return signals
