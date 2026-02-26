from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 2.25,
         'tp_atr_mult': 5.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ATR stop-loss multiplier': ParameterSpec(
                name='ATR stop-loss multiplier',
                min_val=1.0,
                max_val=4.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'ATR take-profit multiplier': ParameterSpec(
                name='ATR take-profit multiplier',
                min_val=1.0,
                max_val=4.0,
                default=5.0,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=2.0,
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=5.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Generate ONLY the body lines to insert inside generate_signals. Do NOT generate class/imports/function signature. Indicators available in this method: ['atr']
        def generate_signals(indicators):
            # LONG intent: macd.macd crosses above macd.signal and momentum > 0.5 and momentum < 0.8 AND rsi > 70 AND rsi < 30 AND atr > 2*bollinger.upper AND close >= bollinger.upper
            signals_long = []
            for i, row in df.iterrows():
                if (row['macd'] > 0 and abs(row['momentum']) > 0.5 and abs(row['momentum']) < 0.8) \
                        and row['rsi'] > 70 and row['rsi'] < 30 \
                        and row['atr'] >= 2 * indicators['bollinger']['upper'][i] \
                        and row['close'] >= indicators['bollinger']['upper'][i]:
                    signals_long.append(1.0)
                else:
                    signals_long.append(-1.0)
            # SHORT intent: macd.macd crosses below macd.signal and momentum > -0.5 and momentum < -0.8 AND rsi < 10 AND rsi > 90 AND atr <= 2*bollinger.lower AND close <= bollinger.lower
            signals_short = []
            for i, row in df.iterrows():
                if (row['macd'] < 0 and abs(row['momentum']) > -0.5) \
                        and abs(row['momentum']) < -0.8 \
                        and row['rsi'] < 10 or row['rsi'] > 90 \
                        and row['atr'] <= 2 * indicators['bollinger']['lower'][i] \
                        and row['close'] <= indicators['bollinger']['lower'][i]:
                    signals_short.append(-1.0)
                else:
                    signals_short.append(1.0)
            # Create the final output dataframe with LONG/SHORT signal values
        return signals
