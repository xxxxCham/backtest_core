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
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)
            n = len(df)

            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # Implement explicit LONG / SHORT / FLAT logic
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            if "rsi" in indicators and "ema" in indicators and "atr" in indicators:
                rsi_window, ema_window, atr_window = int(indicators['rsi']), \
                                                    int(indicators['ema']), \
                                                    int(indicators['atr'])

                # RSI calculation here...
                # EMA calculation here...
                # ATR calculation here...

            entry_mask = np.zeros((len(df)), dtype=bool)
            exit_mask = np.zeros((len(df)), dtype=bool)

            if "bb_stop_long" in df and "bb_tp_long" in df:
                atr = np.nan_to_num(indicators['atr'])

                # Implement ATR based stop loss and take profit logic here...

                entry_mask[df['close'].between(df['bb_stop_long'] - 2*atr, df['bb_stop_long'] + 2*atr)] = 1.0
                exit_mask[df['close'] > df['bb_tp_long']] = 1.0

            signals[entry_mask | exit_mask] += (entry_mask - exit_mask) * 0.5

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
