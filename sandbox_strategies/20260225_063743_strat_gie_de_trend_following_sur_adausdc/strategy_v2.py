from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_aroon_adx_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'aroon', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'aroon_period': 14,
         'atr_period': 14,
         'ichimoku_kijun_period': 26,
         'ichimoku_senkou_span_b_period': 52,
         'ichimoku_tenkan_period': 9,
         'leverage': 1,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 3.0,
         'warmup': 60}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ichimoku_tenkan_period': ParameterSpec(
                name='ichimoku_tenkan_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'ichimoku_kijun_period': ParameterSpec(
                name='ichimoku_kijun_period',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'ichimoku_senkou_span_b_period': ParameterSpec(
                name='ichimoku_senkou_span_b_period',
                min_val=30,
                max_val=100,
                default=52,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=30,
                max_val=120,
                default=60,
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
        # Initialize signals already done by caller; ensure warmup is flat
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract price series
        close = df["close"].values

        # Extract and sanitize indicators
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])

        ar = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(ar["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(ar["aroon_down"])

        adx_dict = indicators['adx']
        adx_val = np.nan_to_num(adx_dict["adx"])

        atr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_entry = (
            (close > senkou_a) &
            (close > senkou_b) &
            (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) &
            (adx_val > 25)
        )

        short_entry = (
            (close < senkou_a) &
            (close < senkou_b) &
            (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) &
            (adx_val > 25)
        )

        # Exit conditions: cross below senkou_a, cross above senkou_b, or weak ADX
        prev_close = np.roll(close, 1)
        prev_senkou_a = np.roll(senkou_a, 1)
        prev_senkou_b = np.roll(senkou_b, 1)
        prev_close[0] = np.nan
        prev_senkou_a[0] = np.nan
        prev_senkou_b[0] = np.nan

        cross_down_a = (close < senkou_a) & (prev_close >= prev_senkou_a)
        cross_up_b = (close > senkou_b) & (prev_close <= prev_senkou_b)

        exit_mask = cross_down_a | cross_up_b | (adx_val < 20)

        # Prevent entries on exit bars
        long_entry = long_entry & ~exit_mask
        short_entry = short_entry & ~exit_mask

        # Apply masks to long_mask / short_mask
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Set signal values
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0  # explicit flat on exit bars

        # ATR‑based stop‑loss and take‑profit
        stop_mult = float(params.get("stop_atr_mult", 1.3))
        tp_mult = float(params.get("tp_atr_mult", 3.0))

        # Initialize SL/TP columns with NaN to avoid leftover values
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entries
        if long_mask.any():
            entry_price_long = close[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_mult * atr[long_mask]

        # Short entries
        if short_mask.any():
            entry_price_short = close[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
