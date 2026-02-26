from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_vortex_atr_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'ichimoku_kijun_period': 26,
            'ichimoku_senkou_span_b_period': 52,
            'ichimoku_tenkan_period': 9,
            'leverage': 1,
            'stop_atr_mult': 1.3,
            'tp_atr_mult': 3.0,
            'vortex_period': 14,
            'warmup': 60,
        }

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
            'vortex_period': ParameterSpec(
                name='vortex_period',
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
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        # initialise output series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        n = len(df)
        warmup = int(params.get('warmup', 50))

        # price series
        close = df['close']

        # extract required indicator arrays
        ichimoku = indicators['ichimoku']
        vortex = indicators['vortex']

        # long entry: price above both Senkou A & B and Vortex up > down
        long_mask = (
            (close > indicators['ichimoku']["senkou_a"])
            & (close > indicators['ichimoku']["senkou_b"])
            & (indicators['vortex']["vi_plus"] > indicators['vortex']["vi_minus"])
        )

        # short entry: price below both Senkou A & B and Vortex down > up
        short_mask = (
            (close < indicators['ichimoku']["senkou_a"])
            & (close < indicators['ichimoku']["senkou_b"])
            & (indicators['vortex']["vi_minus"] > indicators['vortex']["vi_plus"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # enforce warm‑up period where no signals are generated
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals