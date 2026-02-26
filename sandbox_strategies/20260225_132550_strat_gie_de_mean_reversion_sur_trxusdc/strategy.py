from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_bollinger_cci_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'cci', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'cci_threshold': 70,
         'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_threshold': ParameterSpec(
                name='cci_threshold',
                min_val=50,
                max_val=100,
                default=70,
                param_type='int',
                step=1,
            ),
            'rsi_oversold': ParameterSpec(
                name='rsi_oversold',
                min_val=10,
                max_val=40,
                default=30,
                param_type='int',
                step=1,
            ),
            'rsi_overbought': ParameterSpec(
                name='rsi_overbought',
                min_val=60,
                max_val=90,
                default=70,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=50,
                default=20,
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

        # Extract indicator arrays with nan_to_num
        close = df["close"].values
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        cci = np.nan_to_num(indicators['cci'])
        rsi = np.nan_to_num(indicators['rsi'])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        cci_thresh = params.get("cci_threshold", 70)
        rsi_ovb = params.get("rsi_overbought", 70)
        rsi_ovs = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Long entry: close <= lower AND cci <= -threshold AND rsi < oversold
        long_mask = (close <= lower) & (cci <= -cci_thresh) & (rsi < rsi_ovs)

        # Short entry: close >= upper AND cci >= threshold AND rsi > overbought
        short_mask = (close >= upper) & (cci >= cci_thresh) & (rsi > rsi_ovb)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: long exits when close crosses middle or rsi > 50
        # short exits when close crosses middle downward or rsi < 50
        # Use simple crossing detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_up_middle = (close > middle) & (prev_close <= middle)
        cross_down_middle = (close < middle) & (prev_close >= middle)

        long_exit_mask = cross_up_middle | (rsi > 50)
        short_exit_mask = cross_down_middle | (rsi < 50)

        # Clear signals on exit bars
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # Warmup protection: first 50 bars flat
        signals.iloc[:50] = 0.0

        # Prepare ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry levels
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short entry levels
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
