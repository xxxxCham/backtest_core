from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_improved')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 2, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
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
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        stop_atr_mult = float(params.get("stop_atr_mult", 1.25))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        donchian = indicators['donchian']
        adx = indicators['adx']['adx']
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = np.nan_to_num(indicators['atr'])

        long_mask = (close > indicators['donchian']["upper"]) & (adx > 35)
        short_mask = (close < indicators['donchian']["lower"]) & (adx > 20)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down = (close < indicators['donchian']["middle"]) & (prev_close >= indicators['donchian']["middle"])
        cross_up = (close > indicators['donchian']["middle"]) & (prev_close <= indicators['donchian']["middle"])
        cross_any = cross_down | cross_up
        exit_mask = cross_any | (adx < 20)
        signals[exit_mask] = 0.0

        df.loc[:, "bb_stop_long"] = close - stop_atr_mult * atr
        df.loc[:, "bb_tp_long"] = close + tp_atr_mult * atr
        signals.iloc[:warmup] = 0.0
        return signals
