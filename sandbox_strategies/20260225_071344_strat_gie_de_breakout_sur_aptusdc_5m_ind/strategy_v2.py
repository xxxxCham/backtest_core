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
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.84,
         'volume_osc_fast': 14,
         'volume_osc_slow': 28,
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
            'volume_osc_fast': ParameterSpec(
                name='volume_osc_fast',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'volume_osc_slow': ParameterSpec(
                name='volume_osc_slow',
                min_val=10,
                max_val=60,
                default=28,
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
        # Initialize signals and masks
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays (wrapped with nan_to_num)
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])

        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        cloud_pos = np.nan_to_num(ich["cloud_position"])

        # ATR SMA (simple moving average)
        atr_period = int(params.get("atr_period", 14))
        atr_sma = np.convolve(atr, np.ones(atr_period) / atr_period, mode="same")

        # Entry conditions
        long_mask = (close > senkou_a) & (atr > atr_sma) & (vol_osc > 0)
        short_mask = (close < senkou_b) & (atr > atr_sma) & (vol_osc < 0)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare SL/TP columns (initialize with NaN)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR‑based stop‑loss and take‑profit levels
        stop_atr_mult = params.get("stop_atr_mult", 1.6)
        tp_atr_mult = params.get("tp_atr_mult", 3.84)

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
