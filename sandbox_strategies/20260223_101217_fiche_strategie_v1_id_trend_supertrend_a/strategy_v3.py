from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_ema_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.25, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_atr_period': ParameterSpec(
                name='supertrend_atr_period',
                min_val=10,
                max_val=30,
                default=19,
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
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=17,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
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
                default=2.25,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        # Extract raw indicator arrays
        direction_raw = indicators['supertrend']["direction"]
        adx_raw = indicators['adx']["adx"]
        ema_raw = np.nan_to_num(indicators['ema'])
        atr_raw = np.nan_to_num(indicators['atr'])
        close_raw = df["close"].values

        # 20‑period ATR moving average using convolution
        atr_ma20 = np.convolve(atr_raw, np.ones(20) / 20, mode="same")

        # Validity mask to avoid NaN propagation
        valid = (
            ~np.isnan(direction_raw)
            & ~np.isnan(adx_raw)
            & ~np.isnan(ema_raw)
            & ~np.isnan(atr_raw)
            & ~np.isnan(close_raw)
        )

        # Entry conditions
        long_mask = (
            valid
            & (direction_raw == 1)
            & (adx_raw > 25)
            & (close_raw > ema_raw)
            & (atr_raw > atr_ma20)
        )
        short_mask = (
            valid
            & (direction_raw == -1)
            & (adx_raw > 25)
            & (close_raw < ema_raw)
            & (atr_raw > atr_ma20)
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop‑loss and take‑profit levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = (
            close_raw[entry_long] - params["stop_atr_mult"] * atr_raw[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close_raw[entry_long] + params["tp_atr_mult"] * atr_raw[entry_long]
        )

        df.loc[entry_short, "bb_stop_short"] = (
            close_raw[entry_short] + params["stop_atr_mult"] * atr_raw[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close_raw[entry_short] - params["tp_atr_mult"] * atr_raw[entry_short]
        )
        signals.iloc[:warmup] = 0.0
        return signals
