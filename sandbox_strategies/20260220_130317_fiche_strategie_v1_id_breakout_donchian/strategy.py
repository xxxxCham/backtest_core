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
                signals.iloc[:warmup] = 0.0

                rsi = np.nan_to_num(indicators['rsi'])
                ema = np.nan_to_num(indicators['ema'])
                atr = np.nan_to_num(indicators['atr'])
                close = df["close"].values

                # Momentum confirmation: RSI not extreme
                rsi_momentum = (rsi < 70) & (rsi > 30)

                # Long entry: bullish momentum + price above EMA
                long_condition = (
                    close > ema &  # Price above EMA
                    rsi_momentum &  # Healthy momentum
                    (ema != 0)  # Avoid division by zero
                )
                long_mask = np.zeros(n, dtype=bool)
                long_mask[warmup:] = long_condition[warmup:]

                # Short entry: bearish momentum + price below EMA
                short_condition = (
                    close < ema &  # Price below EMA
                    rsi_momentum &  # Healthy momentum
                    (ema != 0)  # Avoid division by zero
                )
                short_mask = np.zeros(n, dtype=bool)
                short_mask[warmup:] = short_condition[warmup:]

                # Apply signals
                signals[long_mask] = 1.0
                signals[short_mask] = -1.0

                # Exit conditions
                momentum_invalidated = ~rsi_momentum
                opposite_signal = (signals != 0) & (
                    (signals == 1.0) & short_mask | 
                    (signals == -1.0) & long_mask
                )

                exit_condition = opposite_signal | momentum_invalidated
                exit_condition = np.zeros(n, dtype=bool)
                exit_condition[warmup:] = exit_condition[0] | exit_condition[warmup:]

                signals[exit_condition] = 0.0

                # ATR-based risk management
                df.loc[:, "sl_level"] = np.nan
                df.loc[:, "tp_level"] = np.nan

                atr_values = np.zeros(n, dtype=float)
                atr_values[warmup:] = atr[warmup:]

                entry_prices = close.copy()
                entry_prices[:warmup] = np.nan

                df.loc[:, "sl_level"] = entry_prices - 1.5 * atr_values
                df.loc[:, "tp_level"] = entry_prices + 3.0 * atr_values

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
