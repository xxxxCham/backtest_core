from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_mfi_macd_atr_momentum_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_vol_threshold': 0.0005,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'mfi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=20,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'mfi_period': ParameterSpec(
                name='mfi_period',
                min_val=5,
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
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
                default=3.6,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=5,
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
        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators with NaN handling
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # Previous values for momentum / divergence checks
        prev_mfi = np.roll(mfi, 1)
        prev_mfi[0] = np.nan
        prev_macd_hist = np.roll(macd_hist, 1)
        prev_macd_hist[0] = np.nan

        # Volatility filter
        atr_vol_threshold = float(params.get("atr_vol_threshold", 0.0005))
        vol_mask = atr > atr_vol_threshold

        # Long entry: bullish histogram, MFI rising above 50, ATR filter
        long_mask = (
            (macd_hist > 0)
            & (mfi > 50)
            & (mfi > prev_mfi)
            & vol_mask
        )

        # Short entry: bearish histogram, MFI falling below 50, ATR filter
        short_mask = (
            (macd_hist < 0)
            & (mfi < 50)
            & (mfi < prev_mfi)
            & vol_mask
        )

        # Exit condition: MACD histogram crosses zero OR MFI crosses 50
        macd_cross_up = (macd_hist > 0) & (prev_macd_hist <= 0)
        macd_cross_down = (macd_hist < 0) & (prev_macd_hist >= 0)
        macd_cross = macd_cross_up | macd_cross_down

        mfi_cross_up = (mfi > 50) & (np.roll(mfi, 1) <= 50)
        mfi_cross_down = (mfi < 50) & (np.roll(mfi, 1) >= 50)
        mfi_cross = mfi_cross_up | mfi_cross_down

        exit_mask = macd_cross | mfi_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0  # ensure flat on exit bars

        # Prepare SL/TP columns (ATR‑based)
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        close = df["close"].values
        stop_mult = float(params.get("stop_atr_mult", 2.0))
        tp_mult = float(params.get("tp_atr_mult", 3.6))

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
