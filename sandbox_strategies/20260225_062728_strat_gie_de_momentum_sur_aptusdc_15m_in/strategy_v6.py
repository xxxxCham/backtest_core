from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_rsi_ema_atr_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'atr_vol_threshold': 0.0005,
            'ema_period': 50,
            'leverage': 1,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_period': 14,
            'stop_atr_mult': 2.0,
            'tp_atr_mult': 3.6,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=20,
                max_val=200,
                default=50,
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
                min_val=0.5,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate long/short signals based on RSI, EMA and ATR filters.
        Long  : RSI > 55, close > EMA, ATR > atr_vol_threshold
        Short : RSI < 45, close < EMA, ATR > atr_vol_threshold
        """
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Retrieve indicator arrays
        rsi = indicators['rsi']          # numpy array
        ema = indicators['ema']          # numpy array
        atr = indicators['atr']          # numpy array

        # Parameters
        atr_vol_threshold = float(params.get('atr_vol_threshold', 0.0005))
        warmup = int(params.get('warmup', 50))

        # Build boolean masks
        long_mask = (rsi > 55) & (df['close'].values > ema) & (atr > atr_vol_threshold)
        short_mask = (rsi < 45) & (df['close'].values < ema) & (atr > atr_vol_threshold)

        # Apply masks to signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Zero out signals during warm‑up period
        signals.iloc[:warmup] = 0.0

        return signals