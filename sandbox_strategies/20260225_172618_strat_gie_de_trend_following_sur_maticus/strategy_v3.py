from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_aroon_atr_trend_follow')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_threshold': 70,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.9,
         'tp_atr_mult': 3.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_threshold': ParameterSpec(
                name='aroon_threshold',
                min_val=50,
                max_val=100,
                default=70,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.9,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.2,
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

        close = np.nan_to_num(df["close"].values)

        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        upper_cloud = np.maximum(senkou_a, senkou_b)
        lower_cloud = np.minimum(senkou_a, senkou_b)
        middle_cloud = (senkou_a + senkou_b) / 2.0

        aroon = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        aroon_threshold = float(params.get("aroon_threshold", 70.0))

        # Cross helpers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper_cloud, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower_cloud, 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(middle_cloud, 1)
        prev_middle[0] = np.nan

        cross_up_upper = (close > upper_cloud) & (prev_close <= prev_upper)
        cross_down_lower = (close < lower_cloud) & (prev_close >= prev_lower)

        # Long entry
        long_mask = (
            cross_up_upper
            & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down'])
            & (indicators['aroon']['aroon_up'] > aroon_threshold)
        )

        # Short entry
        short_mask = (
            cross_down_lower
            & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up'])
            & (indicators['aroon']['aroon_down'] > aroon_threshold)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR based SL/TP levels
        atr = np.nan_to_num(indicators['atr'])
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.9))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.2))

        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
