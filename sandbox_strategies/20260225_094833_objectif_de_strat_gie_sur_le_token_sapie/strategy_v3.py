from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='Snake Case Name')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr']

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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Write SL/TP columns into df if using ATR-based risk management
        if "sl_level" in df and "tp_level" in df:
            sl_mult = params.get("stoploss_multiplier", 1)
            tp_mult = params.get("takeprofit_multiplier", 1)

            # Initialize SL/TP columns with NaN (no level = no stop)
            df["sl_level"] = np.nan * np.ones(n, dtype=np.float64)
            df["tp_level"] = np.nan * np.ones(n, dtype=np.float64)

            # Compute SL/TP levels based on ATR-based risk management parameters 
            atr = indicators['atr']
            sl_level = df['close'].rolling('{warmup}d'.format(warmup=warmup), min_periods=1).apply(lambda x: np.mean(x) * sl_mult, raw=True)[-1]  # noqa
            tp_level = df['close'].rolling('{warmup}d'.format(warmup=warmup), min_periods=1).apply(lambda x: np.mean(x) * tp_mult, raw=True)[-1]  # noqa

        # Implement explicit LONG / SHORT / FLAT logic here
        signals.iloc[:warmup] = 0.0
        return signals
