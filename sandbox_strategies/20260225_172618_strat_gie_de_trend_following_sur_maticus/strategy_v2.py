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
            'aroon_threshold': ParameterSpec(
                name='aroon_threshold',
                min_val=50,
                max_val=100,
                default=70,
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

        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        close = df["close"].values

        upper_cloud = np.maximum(senkou_a, senkou_b)
        lower_cloud = np.minimum(senkou_a, senkou_b)

        ar = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(ar["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(ar["aroon_down"])

        atr = np.nan_to_num(indicators['atr'])

        # Helper cross functions
        def cross_up(x, y):
            px = np.roll(x, 1); py = np.roll(y, 1)
            px[0] = np.nan; py[0] = np.nan
            return (x > y) & (px <= py)

        def cross_down(x, y):
            px = np.roll(x, 1); py = np.roll(y, 1)
            px[0] = np.nan; py[0] = np.nan
            return (x < y) & (px >= py)

        # Long entry: close crosses above upper cloud and Aroon up > down & > threshold
        long_mask = (
            cross_up(close, upper_cloud)
            & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down'])
            & (indicators['aroon']['aroon_up'] > params.get("aroon_threshold", 70))
        )

        # Short entry: close crosses below lower cloud and Aroon down > up & > threshold
        short_mask = (
            cross_down(close, lower_cloud)
            & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up'])
            & (indicators['aroon']['aroon_down'] > params.get("aroon_threshold", 70))
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long = (
            cross_down(close, upper_cloud)
            | (indicators['aroon']['aroon_up'] < indicators['aroon']['aroon_down'])
            | (indicators['aroon']['aroon_up'] < 50)
        )
        exit_short = (
            cross_up(close, lower_cloud)
            | (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up'])
            | (indicators['aroon']['aroon_down'] < 50)
        )

        # Apply exits: set to 0 where exit conditions met after entry
        # For simplicity, we allow exit signals to overwrite earlier signals
        # but not create new entries on the same bar
        # We'll use a simple approach: any exit condition sets signal to 0
        # where current signal is non-zero
        signals[(signals == 1.0) & exit_long] = 0.0
        signals[(signals == -1.0) & exit_short] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.9)
        tp_atr_mult = params.get("tp_atr_mult", 3.2)

        long_entry_mask = signals == 1.0
        short_entry_mask = signals == -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = (
            close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        )
        df.loc[long_entry_mask, "bb_tp_long"] = (
            close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]
        )
        df.loc[short_entry_mask, "bb_stop_short"] = (
            close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        )
        df.loc[short_entry_mask, "bb_tp_short"] = (
            close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        )
        signals.iloc[:warmup] = 0.0
        return signals
