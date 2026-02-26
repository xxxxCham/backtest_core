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
         'tp_atr_mult': 1.9,
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
                max_val=4.0,
                default=1.9,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
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
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        sma = np.nan_to_num(indicators['sma'])
        vip_plus = np.nan_to_num(indicators['vortex']["vi_plus"])
        vip_minus = np.nan_to_num(indicators['vortex']["vi_minus"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Helper cross functions
        def cross_up(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x, y):
            px = np.roll(x, 1)
            py = np.roll(y, 1)
            px[0] = np.nan
            py[0] = np.nan
            return (x < y) & (px >= py)

        # Entry conditions
        long_cross = cross_up(indicators['aroon']['aroon_up'], indicators['aroon']['aroon_down'])
        short_cross = cross_up(indicators['aroon']['aroon_down'], indicators['aroon']['aroon_up'])

        long_mask = long_cross & (close > sma) & (vip_plus > vip_minus)
        short_mask = short_cross & (close < sma) & (vip_minus > vip_plus)

        # Exit conditions
        exit_mask = cross_down(indicators['aroon']['aroon_up'], indicators['aroon']['aroon_down']) | (vip_minus > vip_plus)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP for long entries
        stop_atr_mult = params.get("stop_atr_mult", 1.8)
        tp_atr_mult = params.get("tp_atr_mult", 1.9)
        df.loc[signals == 1.0, "bb_stop_long"] = close[signals == 1.0] - stop_atr_mult * atr[signals == 1.0]
        df.loc[signals == 1.0, "bb_tp_long"] = close[signals == 1.0] + tp_atr_mult * atr[signals == 1.0]

        # ATR-based SL/TP for short entries
        df.loc[signals == -1.0, "bb_stop_short"] = close[signals == -1.0] + stop_atr_mult * atr[signals == -1.0]
        df.loc[signals == -1.0, "bb_tp_short"] = close[signals == -1.0] - tp_atr_mult * atr[signals == -1.0]
        signals.iloc[:warmup] = 0.0
        return signals
