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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        close = df["close"].values
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        atr = np.nan_to_num(indicators['atr'])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])

        # ATR SMA for volatility expansion filter
        atr_period = int(params.get("atr_period", 14))
        if atr_period > 1:
            kernel = np.ones(atr_period) / atr_period
            atr_sma = np.convolve(atr, kernel, mode="same")
        else:
            atr_sma = atr.copy()

        # Previous close for breakout detection
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        # Long entry conditions
        long_cond = (
            (close > senkou_a) &
            (atr > atr_sma) &
            (vol_osc > 0) &
            (close > prev_close + atr) &
            (~np.isnan(prev_close))
        )
        long_mask = long_cond

        # Short entry conditions
        short_cond = (
            (close < senkou_b) &
            (atr > atr_sma) &
            (vol_osc < 0) &
            (close < prev_close - atr) &
            (~np.isnan(prev_close))
        )
        short_mask = short_cond

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop‑loss and take‑profit levels
        stop_mult = float(params.get("stop_atr_mult", 1.6))
        tp_mult = float(params.get("tp_atr_mult", 3.84))

        # Initialize SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Long SL/TP
        if long_mask.any():
            entry_price_long = close[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_mult * atr[long_mask]

        # Short SL/TP
        if short_mask.any():
            entry_price_short = close[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
