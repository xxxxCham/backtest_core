from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Phase Lock Proposer')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'momentum']

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
        def generate_signals(indicators, bars):
            # Default parameters
            default_params = {'leverage': 1}

            # Generate signals
            df = pd.DataFrame() if 'df' not in locals() else df
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            for _, row in bars.iterrows():
                bar_close = row['close']

                # Phase-locked oscillator
                pl_oscillator = np.where((bar_close > indicators['bollinger']['middle']) & (indicators['momentum'][0] > 50), 1, 0)

                # Momentum
                momentum = np.where(np.abs(indicators['momentum'] - 50) < 50, 1, 0)

                signals[int(row['_index'])] = (pl_oscillator * row['close']) + ((~pl_oscillator & momentum) * bar_close)

            # Calculate Donchian breakout
            indicators['donchian']['upper'] = np.roll(indicators['donchian']['upper'], 1)

            signals *= (bar_close < indicators['donchian']['upper'][int(row['_index'])]) & (bar_close > indicators['donchian']['lower'][int(row['_index'])])

            # Calculate ATR-based SL/TP for long and short positions
            if 'df' in locals():
                df = pd.DataFrame(data=signals, index=df.index)

                # Long position - Stop Loss is based on ADR (Average Daily Range)
                df['bb_stop_long'] = df['close'].rolling('5d').mean().add(indicators['donchian']['upper'], 1).subtract(indicators['donchian']['lower'], 1) * default_params['leverage']

                # Long position - Take Profit is based on ADR (Average Daily Range)
                df['bb_tp_long'] = df['close'].rolling('5d').mean().add(df['bb_stop_long'], 1).subtract(indicators['donchian']['lower'], 1) * default_params['leverage']

            # Short position - Stop Loss is based on ADR (Average Daily Range)
            df['bb_stop_short'] = df['close'].rolling('5d').mean().add(indicators['donchian']['upper'], 1).subtract(indicators['donchian']['lower'], 1) * default_params['leverage']

            # Short position - Take Profit is based on ADR (Average Daily Range)
            df['bb_tp_short'] = df['close'].rolling('5d').mean().add(df['bb_stop_short'], 1).subtract(indicators['donchian']['lower'], 1) * default_params['leverage']

            return signals, df
        signals.iloc[:warmup] = 0.0
        return signals
