from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='matic_usdc_ichimoku_aroon_atr_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
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
                min_val=1.5,
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

        # helper cross functions
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

        # extract indicator arrays
        close = df["close"].values
        ich = indicators['ichimoku']
        tenkan = np.nan_to_num(ich["tenkan"])
        kijun = np.nan_to_num(ich["kijun"])
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        aro = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(aro["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(aro["aroon_down"])

        atr = np.nan_to_num(indicators['atr'])

        # Long entry conditions
        long_cross = cross_up(close, tenkan)
        long_cond = long_cross & (close > kijun) & (close > senkou_a) & (indicators['aroon']['aroon_up'] > 70)
        long_mask = long_cond

        # Short entry conditions
        short_cross = cross_down(close, tenkan)
        short_cond = short_cross & (close < kijun) & (close < senkou_a) & (indicators['aroon']['aroon_down'] > 70)
        short_mask = short_cond

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions (set to flat)
        exit_cross1 = cross_down(close, senkou_a)
        exit_cross2 = cross_up(close, senkou_b)
        exit_cond = exit_cross1 | exit_cross2 | (indicators['aroon']['aroon_up'] < 30) | (indicators['aroon']['aroon_down'] < 30)
        signals[exit_cond] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.9))
        tp_mult = float(params.get("tp_atr_mult", 3.2))

        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr[entry_long]

        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
