from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_regime_supertrend_bollinger')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'atr_ma_period': 20,
            'atr_period': 14,
            'bollinger_period': 20,
            'bollinger_std_dev': 2.0,
            'leverage': 1,
            'stop_atr_mult': 1.6,
            'supertrend_multiplier': 3.0,
            'supertrend_period': 10,
            'tp_atr_mult': 3.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_ma_period': ParameterSpec(
                name='atr_ma_period',
                min_val=10,
                max_val=50,
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
                default=3.5,
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
                max_val=3,
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
        """Generate long/short signals based on Supertrend, Bollinger and ATR filters."""
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # ---- Compute ATR moving average (atr_ma) ----
        atr_series = pd.Series(indicators['atr'])
        atr_ma_period = int(params.get('atr_ma_period', 20))
        atr_ma = atr_series.rolling(window=atr_ma_period, min_periods=1).mean().values

        # ---- Entry masks -------------------------------------------------------
        long_mask = (
            (df['close'].values > indicators['bollinger']['upper'])
            & (indicators['supertrend']['direction'] == 1)
            & (indicators['atr'] > atr_ma)
        )
        short_mask = (
            (df['close'].values < indicators['bollinger']['lower'])
            & (indicators['supertrend']['direction'] == -1)
            & (indicators['atr'] > atr_ma)
        )

        # ---- Assign signals ------------------------------------------------------
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ---- Optional stop‑loss / take‑profit columns (for downstream use) ----
        # These columns are not used for signal generation but may be useful for
        # later risk‑management steps.
        df.loc[long_mask, 'bb_stop_long'] = df['close'] - indicators['atr'] * params.get('stop_atr_mult', 1.6)
        df.loc[long_mask, 'bb_tp_long'] = df['close'] + indicators['atr'] * params.get('tp_atr_mult', 3.5)
        df.loc[short_mask, 'bb_stop_short'] = df['close'] + indicators['atr'] * params.get('stop_atr_mult', 1.6)
        df.loc[short_mask, 'bb_tp_short'] = df['close'] - indicators['atr'] * params.get('tp_atr_mult', 3.5)

        # ---- Warm‑up period ------------------------------------------------------
        warmup = int(params.get('warmup', 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals