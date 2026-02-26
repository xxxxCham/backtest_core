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
        def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
            signals = np.zeros(len(df), dtype=np.float64)

            # Check if RSI is enabled and set the corresponding value for leverage
            rsi_enabled = indicators['rsi'] in df.columns
            if rsi_enabled:
                rsi_param = params.get('rsi', 14)
                signals *= np.where(df[df['close']].rolling(window=rsi_param).mean() > df[df['close']].rolling(window=rsi_param).std(), 1, 0)

            # Check if EMA is enabled and set the corresponding value for leverage
            ema_enabled = indicators['ema'] in df.columns
            if ema_enabled:
                ema_param = params.get('ema', 26)
                signals *= np.where(df[df['close']].ewm(span=ema_param, min_periods=ema_param//10).mean() > df[df['close']].rolling(window=5).mean(), 1, 0)

            # Check if ATR is enabled and set the corresponding value for leverage
            atr_enabled = indicators['atr'] in df.columns
            if atr_enabled:
                atr_param = params.get('atr', 14)
                signals *= np.where(df[df['close']].rolling(window=atr_param).std() > df[df['close']].rolling(window=atr_param).mean(), 0, 1)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
