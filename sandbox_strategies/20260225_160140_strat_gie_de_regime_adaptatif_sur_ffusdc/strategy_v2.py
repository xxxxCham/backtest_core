from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ffusdc_30m_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 2.0,
         'tp_atr_mult_trend': 4.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=2.0,
                max_val=6.0,
                default=4.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
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

        # Extract indicator arrays
        bb = indicators['bollinger']
        kelt = indicators['keltner']
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        upper_kelt = np.nan_to_num(kelt["upper"])
        lower_kelt = np.nan_to_num(kelt["lower"])
        middle_kelt = np.nan_to_num(kelt["middle"])

        # Entry conditions
        long_mask = (close > upper_bb) & (close > upper_kelt)
        short_mask = (close < lower_bb) & (close < lower_kelt)

        # Exit conditions via cross of close and keltner middle
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_down = (close < middle_kelt) & (prev_close >= middle_kelt)
        cross_up = (close > middle_kelt) & (prev_close <= middle_kelt)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[cross_down] = 0.0
        signals[cross_up] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult_range = float(params.get("tp_atr_mult_range", 2.0))
        tp_atr_mult_trend = float(params.get("tp_atr_mult_trend", 4.2))

        # Long entry levels
        long_entry = signals == 1.0
        if long_entry.any():
            entry_price_long = close[long_entry]
            atr_long = atr[long_entry]
            df.loc[long_entry, "bb_stop_long"] = entry_price_long - stop_atr_mult * atr_long
            trend_mask = entry_price_long > upper_kelt[long_entry]
            tp_mult_long = np.where(trend_mask, tp_atr_mult_trend, tp_atr_mult_range)
            df.loc[long_entry, "bb_tp_long"] = entry_price_long + tp_mult_long * atr_long

        # Short entry levels
        short_entry = signals == -1.0
        if short_entry.any():
            entry_price_short = close[short_entry]
            atr_short = atr[short_entry]
            df.loc[short_entry, "bb_stop_short"] = entry_price_short + stop_atr_mult * atr_short
            trend_mask = entry_price_short < lower_kelt[short_entry]
            tp_mult_short = np.where(trend_mask, tp_atr_mult_trend, tp_atr_mult_range)
            df.loc[short_entry, "bb_tp_short"] = entry_price_short - tp_mult_short * atr_short
        signals.iloc[:warmup] = 0.0
        return signals
