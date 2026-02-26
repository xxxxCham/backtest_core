from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aroon_sma_vortex_trend_follow')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'sma', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'atr_period': 14,
         'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 1.8,
         'tp_atr_mult': 3.42,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.42,
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

        # Extract indicator arrays
        aroon = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        sma = np.nan_to_num(indicators['sma'])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper cross functions
        def cross_up(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x > y) & (prev_x <= prev_y)

        def cross_down(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return (x < y) & (prev_x >= prev_y)

        # Long entry: indicators['aroon']['aroon_up'] crosses above indicators['aroon']['aroon_down'] AND close > sma
        long_mask = cross_up(indicators['aroon']['aroon_up'], indicators['aroon']['aroon_down']) & (close > sma)

        # Short entry: indicators['aroon']['aroon_down'] crosses above indicators['aroon']['aroon_up'] AND close < sma
        short_mask = cross_up(indicators['aroon']['aroon_down'], indicators['aroon']['aroon_up']) & (close < sma)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.8))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.42))

        # Long entry levels
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short entry levels
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
