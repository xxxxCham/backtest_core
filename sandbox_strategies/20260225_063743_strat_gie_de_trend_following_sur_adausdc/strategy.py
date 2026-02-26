from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_aroon_atr_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'atr_period': 14,
         'ichimoku_kijun_period': 26,
         'ichimoku_senkou_span_b_period': 52,
         'ichimoku_tenkan_period': 9,
         'leverage': 1,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 3.0,
         'warmup': 60}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ichimoku_tenkan_period': ParameterSpec(
                name='ichimoku_tenkan_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ichimoku_kijun_period': ParameterSpec(
                name='ichimoku_kijun_period',
                min_val=10,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'ichimoku_senkou_span_b_period': ParameterSpec(
                name='ichimoku_senkou_span_b_period',
                min_val=30,
                max_val=100,
                default=52,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.3,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=30,
                max_val=120,
                default=60,
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
        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Prepare masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract price series
        close = df["close"].values

        # Extract indicators with NaN handling
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        ar = indicators['aroon']
        up = np.nan_to_num(ar["aroon_up"])
        down = np.nan_to_num(ar["aroon_down"])

        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        stop_atr_mult = float(params.get("stop_atr_mult", 1.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))
        warmup = int(params.get("warmup", 50))

        # ----- ENTRY CONDITIONS -----
        # Long entry: price above both cloud lines and strong Aroon up
        long_entry = (
            (close > senkou_a) &
            (close > senkou_b) &
            (up > down) &
            (up >= 70)
        )

        # Short entry: price below both cloud lines and strong Aroon down
        short_entry = (
            (close < senkou_a) &
            (close < senkou_b) &
            (down > up) &
            (down >= 70)
        )

        # ----- EXIT CONDITIONS -----
        # Helper for cross detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_senkou_a = np.roll(senkou_a, 1)
        prev_senkou_a[0] = np.nan

        cross_down = (close < senkou_a) & (prev_close >= prev_senkou_a)  # price crosses below cloud
        cross_up = (close > senkou_a) & (prev_close <= prev_senkou_a)    # price crosses above cloud

        # Long exit: cross below senkou_a OR Aroon up loses strength
        exit_long = cross_down | (up < down)

        # Short exit: cross above senkou_a OR Aroon down loses strength
        exit_short = cross_up | (down < up)

        # Apply exit filters to entry masks
        long_mask = long_entry & ~exit_long
        short_mask = short_entry & ~exit_short

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ----- ATR‑based SL/TP -----
        # Initialize columns with NaN
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
