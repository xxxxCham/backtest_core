from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_regime_keltner_supertrend_bollinger')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'supertrend', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_ma_period': 20,
         'atr_period': 14,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 3.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_mult': ParameterSpec(
                name='keltner_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_ma_period': ParameterSpec(
                name='atr_ma_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=3,
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
        # Extract price series
        close = df["close"].values

        # Extract and sanitize indicators
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])

        st = indicators['supertrend']
        st_dir = np.nan_to_num(st["direction"])

        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])

        atr = np.nan_to_num(indicators['atr'])

        # ATR moving‑average filter
        atr_ma_period = int(params.get("atr_ma_period", 20))
        if atr_ma_period > 1:
            kernel = np.ones(atr_ma_period) / atr_ma_period
            atr_ma = np.convolve(atr, kernel, mode="same")
        else:
            atr_ma = atr

        # Warm‑up protection
        warmup = int(params.get("warmup", 50))

        # Long entry condition
        long_cond = (
            (close > indicators['keltner']['upper'])
            & (st_dir > 0)
            & (close > indicators['bollinger']['upper'])
            & (atr > atr_ma)
        )
        long_cond[:warmup] = False
        long_mask[long_cond] = True

        # Short entry condition
        short_cond = (
            (close < indicators['keltner']['lower'])
            & (st_dir < 0)
            & (close < indicators['bollinger']['lower'])
            & (atr > atr_ma)
        )
        short_cond[:warmup] = False
        short_mask[short_cond] = True

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR‑based stop‑loss and take‑profit levels
        stop_mult = float(params.get("stop_atr_mult", 1.6))
        tp_mult = float(params.get("tp_atr_mult", 3.5))

        # Long positions
        if long_mask.any():
            entry_price_long = close[long_mask]
            atr_long = atr[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_mult * atr_long
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_mult * atr_long

        # Short positions
        if short_mask.any():
            entry_price_short = close[short_mask]
            atr_short = atr[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_mult * atr_short
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_mult * atr_short

        # Return the computed signals
        return signals
        signals.iloc[:warmup] = 0.0
        return signals
