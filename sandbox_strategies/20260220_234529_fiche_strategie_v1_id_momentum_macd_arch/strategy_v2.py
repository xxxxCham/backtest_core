from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='snake_case_name')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
        def generate_signals(self, df, indicators, params):
            n = len(df)

            signals = np.zeros(n, dtype=np.float64)

            if "leverage" in params and "atr" in indicators:
                atr = np.nan_to_num(indicators['atr'])

                # Implement SL/TP columns for ATR-based risk management if specified in default_params
                signals[self.warmup:(n - self.warmup)] = np.where((df['close'] > df['close'].rolling('{}_{}'.format(params['leverage'], params['warmup']))).mean(), 1, signals[self.warmup:(n - self.warmup)]) * \
                    np.where((np.isnan(indicators['rsi'])), 1, signals[self.warmup:(n - self.warmup)])*0.5 + \
                    np.where(((df['close'] < df['close'].rolling('{}_{}'.format(params['leverage'], params['warmup']))).mean()), -1, signals[self.warmup:(n - self.warmup)]-signals[(n - self.warmup):]) * 0.5

                if "sl_level" in params and "tp_level" in params:
                    close = df["close"].values
                    sl = (df["close"] * (params['leverage'] + 1)) / ((params['leverage']) - close.min())*params['sl_level']
                    tp = (df["close"] * (params['leverage'] + 1)) / ((params['leverage']) - close.max())*params['tp_level']

                    signals[self.warmup:(n - self.warmup)] = np.where((sl < df['close'])&(df["close"] > tp), 2, signals[self.warmup:(n - self.warmup)]) * \
                        np.where(((np.isnan(indicators['rsi'])),) & (df["close"]<tp),-2,signals[(n - self.warmup):])*0.5

            return signals[:len(df)-1] # Return all rows except the last one
        signals.iloc[:warmup] = 0.0
        return signals
