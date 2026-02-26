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
        return ['donchian', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 3.0, 'tp_atr_mult': 3.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
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
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        donchian = indicators['donchian']
        upper = np.nan_to_num(indicators['donchian']["upper"])
        lower = np.nan_to_num(indicators['donchian']["lower"])
        adx_data = indicators['adx']
        adx_val = np.nan_to_num(adx_data["adx"])
        close = df["close"].values

        long_mask = (close > upper) & (adx_val > 30)
        short_mask = (close < lower) & (adx_val > 30)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        prev_close = np.roll(close, 1)
        prev_middle = np.nan_to_num(indicators['donchian']["middle"])
        cross_down = (close < prev_middle) & (prev_close >= prev_middle)
        cross_up = (close > prev_middle) & (prev_close <= prev_middle)
        exit_mask = cross_down | cross_up | (adx_val < 15)
        signals[exit_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
