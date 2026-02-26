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
        return {'leverage': 1, 'stop_atr_mult': 2.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        donchian = indicators['donchian']
        adx = np.nan_to_num(indicators['adx']['adx'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        long_mask = (close > indicators['donchian']["upper"]) & (adx > 30)
        short_mask = (close < indicators['donchian']["lower"]) & (adx > 30)
        signals[long_mask] = 1.0
        # signals[short_mask] = -1.0  # uncomment for shorting
        entry_price = np.where(long_mask, close, np.nan)
        stop_atr_mult = params.get('stop_atr_mult', 2.5)
        tp_atr_mult = params.get('tp_atr_mult', 3.0)
        df.loc[:, "bb_stop_long"] = entry_price - stop_atr_mult * atr
        df.loc[:, "bb_tp_long"] = entry_price + tp_atr_mult * atr
        # For short positions: df.loc[:, "bb_stop_short"]  / df.loc[:, "bb_tp_short"]  (uncomment)
        exit_mask = ((close < indicators['donchian']["middle"]) & long_mask) | (adx < 20)
        signals[exit_mask] = 0.0
        return signals
        signals.iloc[:warmup] = 0.0
        return signals
