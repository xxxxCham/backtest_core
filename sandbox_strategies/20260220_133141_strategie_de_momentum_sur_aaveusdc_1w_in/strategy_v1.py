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

                ema10 = np.nan_to_num(indicators['ema'])
                ema30 = np.nan_to_num(indicators['ema'])
                adx_val = np.nan_to_num(indicators['adx']["adx"])
                atr_val = np.nan_to_num(indicators['atr'])

                prev_ema10 = np.roll(ema10, 1)
                prev_ema30 = np.roll(ema30, 1)
                prev_adx = np.roll(adx_val, 1)

                prev_ema10[0] = np.nan
                prev_ema30[0] = np.nan
                prev_adx[0] = np.nan

                cross_up = (ema10 > ema30) & (prev_ema10 <= prev_ema30)
                cross_down = (ema10 < ema30) & (prev_ema10 >= prev_ema30)

                long_mask = cross_up & (adx_val >= 25.0)
                short_mask = cross_down & (adx_val >= 25.0)

                signals[long_mask] = 1.0
                signals[short_mask] = -1.0

                entry_mask_long = (signals == 1.0).copy()
                entry_mask_long &= ~np.roll(signals, 1).fillna(0.0) == 1.0

                entry_mask_short = (signals == -1.0).copy()
                entry_mask_short &= ~np.roll(signals, 1).fillna(0.0) == -1.0

                close_prices = df["close"].values

                atr_values_long = atr_val[entry_mask_long]
                df.loc[entry_mask_long, "sl_level"] = close_prices[entry_mask_long] - 2.0 * atr_values_long
                df.loc[entry_mask_long, "tp_level"] = close_prices[entry_mask_long] + 5.0 * atr_values_long

                atr_values_short = atr_val[entry_mask_short]
                df.loc[entry_mask_short, "sl_level"] = close_prices[entry_mask_short] + 2.0 * atr_values_short
                df.loc[entry_mask_short, "tp_level"] = close_prices[entry_mask_short] - 5.0 * atr_values_short

                return signals
        signals.iloc[:warmup] = 0.0
        return signals
