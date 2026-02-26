from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            # Your logic here to generate the trading signals based on the input data and indicators/parameters.
            # For example, you could use a Bollinger Band strategy with ATR-based stop loss and take profit levels:
            long_mask = df['bb_upper'] - df['close'].rolling(14).mean() > params["bollinger_k"] * (df['close'].rolling(14).std())  # Long signal
            short_mask = df['bb_lower'] - df['close'].rolling(14).mean() < params["bollinger_k"] * (df['close'].rolling(14).std()) & signals[-1] == 0.0  # Short signal

            signals[long_mask | short_mask] = 1.0  # Set signals to 1 for long and short positions when the conditions are met

            sl_tp_levels = df['close'] + params["bollinger_a"] * (df['close'].rolling(14).std()) - params["bollinger_k"] * (df['close'].rolling(14).std()) \
                           - df['close'].shift()  # ATR-based stop loss and take profit levels

            signals[signals.rolling('30d').mean().lt((params["price_threshold"]))] = 0.0  # Drop signals if the price falls below a specified threshold

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
