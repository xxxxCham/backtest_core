from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='arbusdc_momentum_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'volume_oscillator', 'supertrend', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 1.5,
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
            'volume_oscillator_period': ParameterSpec(
                name='volume_oscillator_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1,
                max_val=4,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=3,
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

        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        indicators['supertrend']['direction'] = np.nan_to_num(indicators['supertrend']["direction"])

        long_mask = (indicators['aroon']['aroon_up'] < 20) & (volume_oscillator > 0) & (indicators['supertrend']['direction'] == 1)
        signals[long_mask] = 1.0

        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        short_mask = (indicators['aroon']['aroon_down'] > 80) | (indicators['supertrend']['direction'] == -1)
        signals[short_mask] = -1.0

        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # ATR-based stop-loss and take-profit
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        signals.iloc[:warmup] = 0.0
        return signals