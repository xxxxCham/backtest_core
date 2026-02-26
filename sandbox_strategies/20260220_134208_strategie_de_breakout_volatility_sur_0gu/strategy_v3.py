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

                # Extract and sanitize indicator arrays
                ema = np.nan_to_num(indicators['ema'])
                atr = np.nan_to_num(indicators['atr'])
                rsi = np.nan_to_num(indicators['rsi'])

                # Compute mean ATR and volatility threshold
                atr_mean = np.median(atr)
                high_volatility = atr > np.percentile(atr, 80)

                # Calculate previous values for momentum
                prev_atr = np.roll(atr, 1)
                prev_atr[0] = np.nan
                prev_rsi = np.roll(rsi, 1)
                prev_rsi[0] = np.nan

                # Warmup protection
                warmup = int(params.get("warmup", 50))
                start_idx = warmup

                # Long conditions: price above EMA + momentum + volatility filter
                price = df["close"].values
                above_ema = price > ema
                atr_expansion = atr > 1.5 * atr_mean
                momentum_long = (rsi > 50) & (rsi > prev_rsi)

                long_entry = (
                    above_ema &
                    atr_expansion &
                    momentum_long &
                    ~high_volatility &
                    (np.arange(n) >= warmup)
                )

                # Short conditions: price below EMA + momentum + volatility filter
                below_ema = price < ema
                momentum_short = (rsi < 50) & (rsi < prev_rsi)

                short_entry = (
                    below_ema &
                    atr_expansion &
                    momentum_short &
                    ~high_volatility &
                    (np.arange(n) >= warmup)
                )

                # Signal assignment
                signals[long_entry] = 1.0
                signals[short_entry] = -1.0

                # Risk management: SL/TP levels
                stop_mult = 2.0
                tp_mult = 5.0

                # Long SL/TP
                entry_long = (signals == 1.0) & (np.arange(n) >= warmup)
                df.loc[entry_long, "sl_level"] = price[entry_long] - stop_mult * atr[entry_long]
                df.loc[entry_long, "tp_level"] = price[entry_long] + tp_mult * atr[entry_long]

                # Short SL/TP
                entry_short = (signals == -1.0) & (np.arange(n) >= warmup)
                df.loc[entry_short, "sl_level"] = price[entry_short] + stop_mult * atr[entry_short]
                df.loc[entry_short, "tp_level"] = price[entry_short] - tp_mult * atr[entry_short]

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
