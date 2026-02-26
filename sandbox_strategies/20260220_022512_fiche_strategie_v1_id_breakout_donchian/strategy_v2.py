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
        return {'leverage': 1, 'stop_atr_mult': 1.25, 'tp_atr_mult': 4.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=6.0,
                default=4.5,
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
        close = df['close'].values
        stop_multiplier = params.get("stop_multiplier", 1.25)
        tp_multiplier = params.get("tp_multiplier", 4.5)
        warmup = int(params.get("warmup", 50))
        signals[:warmup] = 0.0

        # Long entry
        long_mask = (close > indicators['donchian']["upper"]) & (indicators['adx']["adx"] > 35)
        df.loc[long_mask, 'bb_stop_long'] = close[long_mask] - stop_multiplier * atr[long_mask]
        df.loc[long_mask, 'bb_tp_long'] = close[long_mask] + tp_multiplier * atr[long_mask]
        signals[long_mask] = 1.0

        # Short entry
        short_mask = (close < indicators['donchian']["lower"]) & (indicators['adx']["adx"] > 35)
        df.loc[short_mask, 'bb_stop_short'] = close[short_mask] + stop_multiplier * atr[short_mask]
        df.loc[short_mask, 'bb_tp_short'] = close[short_mask] - tp_multiplier * atr[short_mask]
        signals[short_mask] = -1.0

        # Exit
        exit_long_mask = (close < indicators['donchian']["middle"]) | (indicators['adx']["adx"] < 15) & (signals == 1.0)
        df.loc[exit_long_mask, 'bb_stop_long'] = np.nan
        df.loc[exit_long_mask, 'bb_tp_long'] = np.nan
        signals[exit_long_mask] = 0.0

        exit_short_mask = (close > indicators['donchian']["middle"]) | (indicators['adx']["adx"] < 15) & (signals == -1.0)
        df.loc[exit_short_mask, 'bb_stop_short'] = np.nan
        df.loc[exit_short_mask, 'bb_tp_short'] = np.nan
        signals[exit_short_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
