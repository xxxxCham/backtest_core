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
        return {'adx_threshold': 25,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=0,
                max_val=100,
                default=25,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=100,
                default=1,
                param_type='int',
                step=1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=0,
                max_val=100,
                default=30,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
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
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Indicator access
        donchian = indicators['donchian']
        adx_val = np.nan_to_num(indicators['adx']['adx'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry long
        long_mask = (close > indicators['donchian']["upper"]) & (adx_val > 25)
        signals[long_mask] = 1.0
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - 1.75 * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + 4.0 * atr[long_mask]

        # Exit long
        signals[(close < indicators['donchian']["middle"]) | (adx_val < 20)] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
