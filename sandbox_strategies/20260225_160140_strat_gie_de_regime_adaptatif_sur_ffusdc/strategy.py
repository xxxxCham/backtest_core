from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ffusdc_30m_boll_kelt_atr_regime')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.0,
         'tp_atr_mult_range': 2.5,
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
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_std_dev': ParameterSpec(
                name='bollinger_std_dev',
                min_val=1,
                max_val=3,
                default=2,
                param_type='float',
                step=0.1,
            ),
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=30,
                default=20,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_trend': ParameterSpec(
                name='tp_atr_mult_trend',
                min_val=2.0,
                max_val=5.0,
                default=4.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult_range': ParameterSpec(
                name='tp_atr_mult_range',
                min_val=1.0,
                max_val=3.5,
                default=2.5,
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

        # Extract indicators with nan_to_num
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])

        kelt = indicators['keltner']
        upper_kelt = np.nan_to_num(kelt["upper"])
        middle_kelt = np.nan_to_num(kelt["middle"])
        lower_kelt = np.nan_to_num(kelt["lower"])

        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (close > upper_bb) & (close > upper_kelt)
        short_mask = (close < lower_bb) & (close < lower_kelt)

        # Exit conditions: cross of close with keltner middle or bollinger middle
        def cross_any(x, y):
            prev_x = np.roll(x, 1)
            prev_y = np.roll(y, 1)
            prev_x[0] = np.nan
            prev_y[0] = np.nan
            return ((x > y) & (prev_x <= prev_y)) | ((x < y) & (prev_x >= prev_y))

        exit_mask = cross_any(close, middle_kelt) | cross_any(close, middle_bb)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Determine TP multiplier based on Keltner width vs ATR
        kelt_width = upper_kelt - lower_kelt
        tp_mult = np.where(
            kelt_width > 2 * atr,
            params.get("tp_atr_mult_trend", 4.2),
            params.get("tp_atr_mult_range", 2.5),
        )

        # Long entry SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult[long_mask] * atr[long_mask]

        # Short entry SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult[short_mask] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
