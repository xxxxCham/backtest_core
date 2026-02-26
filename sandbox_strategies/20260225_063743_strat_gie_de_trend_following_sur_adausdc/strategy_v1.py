from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ichimoku_aroon_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ichimoku', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
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
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # extract needed arrays
        close = df["close"].values
        ich = indicators['ichimoku']
        senkou_a = np.nan_to_num(ich["senkou_a"])
        senkou_b = np.nan_to_num(ich["senkou_b"])
        ar = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(ar["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(ar["aroon_down"])
        atr = np.nan_to_num(indicators['atr'])

        # parameters
        stop_atr_mult = float(params.get("stop_atr_mult", 1.3))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        # entry conditions
        long_entry = (close > senkou_a) & (close > senkou_b) & (indicators['aroon']['aroon_up'] > 70)
        short_entry = (close < senkou_a) & (close < senkou_b) & (indicators['aroon']['aroon_down'] > 70)

        # assign masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # compute SL/TP for entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
