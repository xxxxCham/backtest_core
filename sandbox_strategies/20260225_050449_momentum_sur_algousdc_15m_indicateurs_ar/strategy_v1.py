from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='algo_momentum_supertrend')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'supertrend', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 2.5,
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
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1,
                max_val=5,
                default=3,
                param_type='float',
                step=0.1,
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

        signals.iloc[:warmup] = 0.0

        aroon = indicators['aroon']
        supertrend = indicators['supertrend']
        ema = np.nan_to_num(indicators['ema'])
        close = df["close"].values

        aroon_upper_cross = indicators['aroon']["aroon_up"] > 0
        aroon_lower_cross = indicators['aroon']["aroon_down"] < 0
        indicators['supertrend']['direction'] = indicators['supertrend']["direction"]
        ema_above_close = ema > close

        long_mask = np.logical_and(
            aroon_upper_cross,
            indicators['supertrend']['direction'] == 1,
            ema_above_close
        )

        short_mask = np.logical_and(
            aroon_lower_cross,
            indicators['supertrend']['direction'] == -1,
            ema < close
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        atr = np.nan_to_num(indicators['atr'])
        if np.any(long_mask):
            entry_prices = close[long_mask]
            stop_loss_levels = entry_prices - params["stop_atr_mult"] * atr[long_mask]
            take_profit_levels = entry_prices + params["tp_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_stop_long"] = stop_loss_levels
            df.loc[long_mask, "bb_tp_long"] = take_profit_levels
        signals.iloc[:warmup] = 0.0
        return signals