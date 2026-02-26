from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'ema', 'roc', 'stochastic']

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
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            signals = pd.Series(0.0, index=data.index, dtype=np.float64)

            # Initialize default parameters
            macd, ma_fast, ma_slow, k_sl, k_tp, stop_atr_mult, leverage = self.default_params 

            # Compute MACD and signal lines for each bar
            macds = ma.MACDMovingAverage(data['close'], window_length=12)
            signals['macd'] = macds.ipping_line() - macds.signal_line()

            # Check if the fast line crosses above or below the signal line and generate a long/short signal accordingly
            crossovers = crossing(macds.fast, macds.slow)
            signals[crossovers] = 1.0 - signals[crossovers].diff() / abs(signals[crossovers]) # apply absolute value to avoid directional bias

            # Compute ATR values for stop loss and take profit calculation based on the default parameters
            atr_values = self.default_params['atr'] * (1 + stop_atr_mult) - leverage / 2
            signals[k_sl] = data['close'].rolling(window=3).apply(lambda x: np.mean(x))[:-3:-1]*stop_atr_mult
            signals[k_tp] = data['close'].rolling(window=3).apply(lambda x: -np.mean(x))[-2:] # using the last two bars for take profit levels

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
