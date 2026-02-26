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
        def generate_signals(
                self,
                df: pd.DataFrame,
                indicators: Dict[str, Any],
                params: Dict[str, Any],
            ) -> pd.Series:
                signals = pd.Series(0.0, index=df.index, dtype=np.float64)
                n = len(df)

                warmup = int(params.get("warmup", 50))
                if warmup > 0:
                    signals.iloc[:warmup] = 0.0

                ema = np.nan_to_num(indicators['ema'])
                rsi = np.nan_to_num(indicators['rsi'])
                atr = np.nan_to_num(indicators['atr'])
                close = df["close"].values

                long_conditions = (
                    (close > ema) &
                    (rsi > 65) &
                    (np.diff(close) > 0)
                )
                long_conditions = np.zeros(n, dtype=bool)
                long_conditions[warmup:] = long_conditions[warmup:]

                short_conditions = (
                    (close < ema) &
                    (rsi < 35) &
                    (np.diff(close) < 0)
                )
                short_conditions = np.zeros(n, dtype=bool)
                short_conditions[warmup:] = short_conditions[warmup:]

                long_mask = long_conditions
                short_mask = short_conditions

                signals[long_mask] = 1.0
                signals[short_mask] = -1.0

                df.loc[:, "sl_level"] = np.nan
                df.loc[:, "tp_level"] = np.nan
                entry_mask = (signals != 0.0)
                df.loc[entry_mask, "sl_level"] = close[entry_mask] - params["stop_atr_mult"] * atr[entry_mask]
                df.loc[entry_mask, "tp_level"] = close[entry_mask] + params["tp_atr_mult"] * atr[entry_mask]

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
