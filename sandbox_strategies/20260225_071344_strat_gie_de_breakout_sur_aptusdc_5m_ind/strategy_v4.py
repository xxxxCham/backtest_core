from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_breakout_ichimoku_donchian_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'donchian', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'donchian_period': 20,
            'ichimoku_kijun_period': 26,
            'ichimoku_senkou_span_b_period': 52,
            'ichimoku_tenkan_period': 9,
            'leverage': 1,
            'stop_atr_mult': 1.6,
            'tp_atr_mult': 3.84,
            'warmup': 60,
        }

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
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=50,
                default=20,
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
        """Generate long (+1), short (-1) or flat (0) signals."""
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))

        # price series
        close = df['close'].values

        # indicator arrays
        donchian_upper = indicators['donchian']['upper']
        donchian_lower = indicators['donchian']['lower']
        ichimoku_senkou_a = indicators['ichimoku']['senkou_a']
        ichimoku_senkou_b = indicators['ichimoku']['senkou_b']
        atr = indicators['atr']

        # ATR SMA (simple moving average) – used as volatility filter
        atr_period = int(params.get('atr_period', 14))
        atr_sma = pd.Series(atr).rolling(window=atr_period, min_periods=1).mean().values

        # previous values for cross detection
        prev_close = np.roll(close, 1)
        prev_upper = np.roll(donchian_upper, 1)
        prev_lower = np.roll(donchian_lower, 1)

        # LONG condition: price breaks above Donchian upper, above Ichimoku Senkou A,
        # and ATR is above its SMA (higher volatility)
        long_mask = (
            (prev_close <= prev_upper)
            & (close > donchian_upper)
            & (close > ichimoku_senkou_a)
            & (atr > atr_sma)
        )

        # SHORT condition: price breaks below Donchian lower, below Ichimoku Senkou B,
        # and ATR is above its SMA
        short_mask = (
            (prev_close >= prev_lower)
            & (close < donchian_lower)
            & (close < ichimoku_senkou_b)
            & (atr > atr_sma)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out initial warm‑up period
        signals.iloc[:warmup] = 0.0

        return signals