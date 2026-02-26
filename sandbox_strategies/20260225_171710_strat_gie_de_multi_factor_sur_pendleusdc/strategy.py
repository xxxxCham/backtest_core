from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rsi_obv_atr_volatility_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'obv', 'atr']

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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
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

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Indicator arrays
        rsi = np.nan_to_num(indicators['rsi'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])

        # Previous values for OBV
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan
        obv_increase = (obv > obv_prev) & (~np.isnan(obv_prev))
        obv_decrease = (obv < obv_prev) & (~np.isnan(obv_prev))

        # ATR mean (using overall mean as proxy for 14‑period average)
        atr_mean = np.mean(atr)

        # Long entry: rsi < oversold, OBV rising, ATR above mean
        long_mask = (
            (rsi < params["rsi_oversold"])
            & obv_increase
            & (atr > atr_mean)
        )

        # Short entry: rsi > overbought, OBV falling, ATR above mean
        short_mask = (
            (rsi > params["rsi_overbought"])
            & obv_decrease
            & (atr > atr_mean)
        )

        # Exit logic: RSI crosses 50 or OBV changes direction
        prev_rsi = np.roll(rsi, 1)
        prev_rsi[0] = np.nan
        cross_rsi_up = (rsi > 50) & (prev_rsi <= 50)
        cross_rsi_down = (rsi < 50) & (prev_rsi >= 50)
        cross_any_rsi = cross_rsi_up | cross_rsi_down

        obv_diff = np.diff(obv)
        obv_diff = np.insert(obv_diff, 0, 0.0)
        obv_sign = np.sign(obv_diff)
        prev_sign = np.roll(obv_sign, 1)
        prev_sign[0] = 0
        direction_change = (
            (obv_sign != prev_sign)
            & (obv_sign != 0)
            & (prev_sign != 0)
        )

        exit_mask = cross_any_rsi | direction_change

        # Apply masks to signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
