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
        return ['donchian', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.25, 'tp_atr_mult': 4.5, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.75,
                max_val=3.0,
                default=1.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=4.0,
                max_val=8.0,
                default=4.5,
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

            # implement ATR-based risk management if present in params
            stop_sl_mult = params.get("stop_sl_mult", 1.5)
            atr = indicators['atr']
            k_sl = params.get("k_sl", 2.0)

            # warmup protection
            n_warmup = int(params.get("warmup", 50))
            df_warmup = df[n + n - n_warmup:]
            close_warmup = df_warmup["close"].values[-n_warmup]

            # implement logic for long entries if applicable
            entry_mask = (close_warmup > close_warmup.rolling(21, min_periods=1).mean()) & \
                         (close_warmup < close_warmup.rolling(21, min_periods=1).std() * k_sl)
            signals[entry_mask] = 1.0

            # implement logic for short entries if applicable
            short_mask = (close_warmup > close_warmup.rolling(21, min_periods=1)) & \
                         (close_warmup < close_warmup.rolling(21, min_periods=1).std() * k_sl)
            signals[short_mask] = -1.0

            return signals
        signals.iloc[:warmup] = 0.0
        return signals
