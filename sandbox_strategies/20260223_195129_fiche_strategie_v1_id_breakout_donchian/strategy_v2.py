from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='FICHE_STRATEIE')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 2.75,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

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
        def generate_signals(df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
            signals = pd.Series(0.0, index=df.index, dtype=np.float64)

            n = len(df)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)

            # Implement explicit LONG / SHORT / FLAT logic
            warmup = int(params.get("warmup", 50))
            signals.iloc[:warmup] = 0.0

            # Write SL/TP columns into df if using ATR-based risk management
            atr = np.nan_to_num(indicators['atr'])
            close = df["close"].values
            entry_mask = (signals == 1.0) & (close > (close[0] + 0.5 * atr))

            signals[entry_mask] = 1.0
            long_mask[entry_mask] = True

            # Find TP level based on ATR
            stop_atr = params.get("stop_loss", {}).get("atrr")
            tp_level = close + (close[-1] - close[0]) * stop_atr / atr  # Assuming that SL is atrr*2

            short_mask |= (signals == -1) & (close < tp_level) & (close > close.mean() - 0.5 * atr)
            signals[short_mask] = -1.0
            long_mask &= ~(short_mask)

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
