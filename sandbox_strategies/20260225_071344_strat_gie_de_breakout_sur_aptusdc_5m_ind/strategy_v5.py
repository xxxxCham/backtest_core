from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_breakout_ichimoku_atr_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'atr', 'volume_oscillator']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ichimoku_kijun_period': 26,
         'ichimoku_senkou_span_b_period': 52,
         'ichimoku_tenkan_period': 9,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.84,
         'volume_oscillator_fast': 5,
         'volume_oscillator_slow': 20,
         'warmup': 60}

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
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'ichimoku_senkou_span_b_period': ParameterSpec(
                name='ichimoku_senkou_span_b_period',
                min_val=30,
                max_val=80,
                default=52,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_fast': ParameterSpec(
                name='volume_oscillator_fast',
                min_val=3,
                max_val=10,
                default=5,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_slow': ParameterSpec(
                name='volume_oscillator_slow',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.84,
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
        # Prepare price and indicator arrays
        close = np.nan_to_num(df["close"].values)

        atr = np.nan_to_num(indicators['atr'])

        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        vol_osc = np.nan_to_num(indicators['volume_oscillator'])

        # Previous close for breakout comparison
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Long entry: price breaks above previous close + ATR, inside Ichimoku cloud, volume up
        long_mask = (
            (close > senkou_a)
            & (close > senkou_b)
            & (close > (prev_close + atr))
            & (vol_osc > 0)
        )

        # Short entry: price breaks below previous close - ATR, below Ichimoku cloud, volume down
        short_mask = (
            (close < senkou_a)
            & (close < senkou_b)
            & (close < (prev_close - atr))
            & (vol_osc < 0)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Ensure warmup region stays flat after signal assignment
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # ATR‑based risk management: write SL/TP levels only on entry bars
        stop_mult = float(params.get("stop_atr_mult", 1.6))
        tp_mult = float(params.get("tp_atr_mult", 3.84))

        # Initialize columns with NaN (optional, pandas will create them on assignment)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long SL/TP
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr[entry_long]

        # Short SL/TP
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr[entry_short]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
