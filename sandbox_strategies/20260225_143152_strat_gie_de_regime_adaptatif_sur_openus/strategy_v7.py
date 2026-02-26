from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adx_keltner_atr_regime')

    @property
    def required_indicators(self) -> List[str]:
        return ['adx', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_threshold': 25,
         'atr_period': 14,
         'keltner_multiplier': 1.5,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'tp_atr_range_mult': 1.5,
         'tp_atr_trend_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=10,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'keltner_multiplier': ParameterSpec(
                name='keltner_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_trend_mult': ParameterSpec(
                name='tp_atr_trend_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_range_mult': ParameterSpec(
                name='tp_atr_range_mult',
                min_val=0.5,
                max_val=3.0,
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

        # Extract indicator arrays safely
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # Entry conditions
        long_mask = (close > indicators['keltner']['upper']) & (adx_val > params["adx_threshold"])
        short_mask = (close < indicators['keltner']['lower']) & (adx_val > params["adx_threshold"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long stop‑loss and take‑profit
        if long_mask.any():
            trend_tp_mult = np.where(
                adx_val[long_mask] > params["adx_threshold"],
                params["tp_atr_trend_mult"],
                params["tp_atr_range_mult"],
            )
            df.loc[long_mask, "bb_stop_long"] = (
                close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            )
            df.loc[long_mask, "bb_tp_long"] = (
                close[long_mask] + trend_tp_mult * atr[long_mask]
            )

        # Short stop‑loss and take‑profit
        if short_mask.any():
            trend_tp_mult = np.where(
                adx_val[short_mask] > params["adx_threshold"],
                params["tp_atr_trend_mult"],
                params["tp_atr_range_mult"],
            )
            df.loc[short_mask, "bb_stop_short"] = (
                close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            )
            df.loc[short_mask, "bb_tp_short"] = (
                close[short_mask] - trend_tp_mult * atr[short_mask]
            )
        signals.iloc[:warmup] = 0.0
        return signals
