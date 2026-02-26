from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='maticusdc_ichimoku_adx_atr_trend_following')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.9, 'tp_atr_mult': 3.2, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.9,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=5.0,
                default=3.2,
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

        close = df["close"].values
        atr_arr = np.nan_to_num(indicators['atr'])

        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])

        # Helper cross functions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_senkou_a = np.roll(senkou_a, 1)
        prev_senkou_a[0] = np.nan
        prev_senkou_b = np.roll(senkou_b, 1)
        prev_senkou_b[0] = np.nan

        cross_up_close_senkou_a = (close > senkou_a) & (prev_close <= prev_senkou_a)
        cross_down_close_senkou_b = (close < senkou_b) & (prev_close >= prev_senkou_b)
        cross_down_close_senkou_a = (close < senkou_a) & (prev_close >= prev_senkou_a)
        cross_up_close_senkou_b = (close > senkou_b) & (prev_close <= prev_senkou_b)

        # Long entry
        long_mask = cross_up_close_senkou_a & (adx_val > 25)
        # Short entry
        short_mask = cross_down_close_senkou_b & (adx_val > 25)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_mask = (
            cross_down_close_senkou_a
            | cross_up_close_senkou_b
            | (adx_val < 20)
        )
        signals[exit_mask] = 0.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_mult = float(params.get("stop_atr_mult", 1.9))
        tp_mult = float(params.get("tp_atr_mult", 3.2))

        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = (
                close[long_mask] - stop_mult * atr_arr[long_mask]
            )
            df.loc[long_mask, "bb_tp_long"] = (
                close[long_mask] + tp_mult * atr_arr[long_mask]
            )
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = (
                close[short_mask] + stop_mult * atr_arr[short_mask]
            )
            df.loc[short_mask, "bb_tp_short"] = (
                close[short_mask] - tp_mult * atr_arr[short_mask]
            )
        signals.iloc[:warmup] = 0.0
        return signals
