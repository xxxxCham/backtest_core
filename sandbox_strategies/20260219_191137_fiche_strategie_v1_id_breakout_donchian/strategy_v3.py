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
        return {'leverage': 1, 'stop_atr_mult': 1.75, 'tp_atr_mult': 6.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=2.0,
                max_val=3.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=4.0,
                max_val=8.0,
                default=6.0,
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
        adx = indicators['adx']
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        long_mask = (close > indicators['donchian']["upper"]) & (indicators['adx']["adx"] > 30) & (prev_close <= prev_close)
        short_mask = (close < indicators['donchian']["lower"]) & (indicators['adx']["adx"] > 30) & (prev_close >= prev_close)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        exit_long_mask = ((close < indicators['donchian']["middle"]) | (indicators['adx']["adx"] < 15)) & (signals == 1.0)
        exit_short_mask = ((close > indicators['donchian']["middle"]) | (indicators['adx']["adx"] < 15)) & (signals == -1.0)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        warmup = int(params.get("warmup", 50))
        signals[:warmup] = 0.0
        stop_atr_mult = params.get("stop_atr_mult", 1.75)
        tp_atr_mult = params.get("tp_atr_mult", 6.0)
        entry_mask = (signals != 0.0)
        df.loc[entry_mask & (signals == 1.0), "bb_stop_long"] = close[entry_mask & (signals == 1.0)] - stop_atr_mult * atr[entry_mask & (signals == 1.0)]
        df.loc[entry_mask & (signals == 1.0), "bb_tp_long"] = close[entry_mask & (signals == 1.0)] + tp_atr_mult * atr[entry_mask & (signals == 1.0)]
        return signals
        signals.iloc[:warmup] = 0.0
        return signals
