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
        return {'adx_period': 16,
         'atr_period': 14,
         'donchian_period': 15,
         'leverage': 1,
         'stop_atr_mult': 2.25,
         'tp_atr_mult': 3.5,
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
                max_val=30,
                default=15,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=1,
                max_val=30,
                default=16,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
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
        donchian = np.nan_to_num(indicators['donchian']['middle'])
        adx_val = np.nan_to_num(indicators['adx']['adx'])
        atr = np.nan_to_num(indicators['atr'])
        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        # Long signal
        long_mask = (close > donchian) & (adx_val > 35)
        signals[long_mask] = 1.0
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - 2.25 * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + 3.5 * atr[long_mask]

        # Short signal (if applicable)
        # short_mask = (close < donchian) & (adx_val > 35)
        # signals[short_mask] = -1.0
        # df.loc[short_mask, "bb_stop_short"] = close[short_mask] + 2.25 * atr[short_mask]
        # df.loc[short_mask, "bb_tp_short"] = close[short_mask] - 3.5 * atr[short_mask]

        # Exit signal
        prev_donchian = np.roll(donchian, 1)
        prev_donchian[0] = np.nan
        exit_long_mask = ((close < prev_donchian) | (adx_val < 25)) & (signals == 1.0)
        signals[exit_long_mask] = 0.0
        # Exit signal for short positions (if applicable)
        # exit_short_mask = ((close > prev_donchian) | (adx_val < 25)) & (signals == -1.0)
        # signals[exit_short_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
