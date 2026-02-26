from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='builder_strategy')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema', 'atr']

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
            n = len(df)
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # implement explicit LONG / SHORT / FLAT logic
            warmup = int(params['warmup']) if 'warmup' in params else 50
            signals[:warmup] = 0.0

            # Write SL/TP columns into df if using ATR-based risk management
            atr = np.nan_to_num(indicators['atr'])
            ema = np.nan_to_num(indicators['ema'])
            close = df['close'].values
            long_mask = ((close > ema) & (close > ema[:-1].rolling(window=14).mean())).astype(np.int8) * 1.0
            short_mask = ((close < ema) & (close < ema[:-1].rolling(window=14).mean())).astype(np.int8) * -1.0

            # For SL/TP entries, compute long stop loss and take profit levels:
            sl_level = close[long_mask] + (close[short_mask] - close[long_mask])*params['sl_ratio']
            tp_level = close[long_mask] - ((close[short_mask] - close[long_mask])*(1+params['tp_ratio']))  # Assuming TP is the highest high after entry

            signals[(df.index > long_mask) & (df.index < short_mask)] = 0.0
            signals[(df.index >= warmup)&(df[short_mask].diff()*signals[-1] <= params['slippage'])] *= -1.0

            return signals[long_mask]   # Replace with your strategy logic here
        signals.iloc[:warmup] = 0.0
        return signals
