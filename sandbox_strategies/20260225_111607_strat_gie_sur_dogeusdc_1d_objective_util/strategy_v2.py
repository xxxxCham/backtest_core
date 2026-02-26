from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='TrendRangeAdaptation')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'donchian', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'donchian_period': 5,
         'ema_length': 20,
         'leverage': 2,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'stop_mult': 1.5,
         'tp_atr_mult': 3.0,
         'tp_mult': 3.0,
         'warmup': 75}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=8,
                max_val=24,
                default=14,
                param_type='int',
                step=1,
            ),
            'ema_length': ParameterSpec(
                name='ema_length',
                min_val=6,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=5,
                max_val=30,
                default=5,
                param_type='int',
                step=1,
            ),
            'stop_mult': ParameterSpec(
                name='stop_mult',
                min_val=1.0,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_mult': ParameterSpec(
                name='tp_mult',
                min_val=2.0,
                max_val=6.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=2,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        def generate_signals(self):
            # Obtain price data series and relevant features from the broker
            price = self.getBroker().getLastPrice()

            macd, macdsignal, macdhist = indicatorbasedstratestyle._MACD(price, timeframe=TimeFrame.Minute14)
            rsi_longterm, _ = indicatorbasedstratestyle._STOCH(price, timeframes=[TimeFrame.Minute60], plot=False)

            bollingerbands_lower, bollingerbands_middle, bollingerbands_upper = ma(price, timeframe=TimeFrame.Minute14, nbdevup=2, nbdevdn=-2)
            donchianchannels_lower, donchianchannels_middle, donchianchannels_upper = simpleorderflow._DONCHIAN(price, timeframes=[7], plot=False)

            # Apply the logic to generate trading signals based on the calculated features and conditions
            buy_condition = macd > 0.15
            sell_condition = macs < -0.35
            long_position = False
        signals.iloc[:warmup] = 0.0
        return signals
