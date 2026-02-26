from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_adx_momentum_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'adx', 'momentum', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'adx_threshold': 25,
         'keltner_multiplier': 2,
         'keltner_period': 20,
         'leverage': 1,
         'momentum_period': 10,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=1.5,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'adx_threshold': ParameterSpec(
                name='adx_threshold',
                min_val=20,
                max_val=35,
                default=25,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=4.0,
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

        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])

        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])

        momentum = np.nan_to_num(indicators['momentum'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        prev_close = np.roll(close, 1)
        prev_kelt_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_kelt_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_kelt_middle = np.roll(indicators['keltner']['middle'], 1)

        prev_close[0] = np.nan
        prev_kelt_upper[0] = np.nan
        prev_kelt_lower[0] = np.nan
        prev_kelt_middle[0] = np.nan

        cross_above_upper = (close > indicators['keltner']['upper']) & (prev_close <= prev_kelt_upper)
        cross_below_lower = (close < indicators['keltner']['lower']) & (prev_close >= prev_kelt_lower)
        cross_middle_up = (close > indicators['keltner']['middle']) & (prev_close <= prev_kelt_middle)
        cross_middle_down = (close < indicators['keltner']['middle']) & (prev_close >= prev_kelt_middle)

        adx_threshold = params.get("adx_threshold", 25)
        stop_atr_mult = params.get("stop_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entry_mask = cross_above_upper & (adx_val > adx_threshold) & (momentum > 0)
        short_entry_mask = cross_below_lower & (adx_val > adx_threshold) & (momentum < 0)

        long_mask = long_entry_mask
        short_mask = short_entry_mask

        exit_long_mask = cross_middle_down | (adx_val < 20)
        exit_short_mask = cross_middle_up | (adx_val < 20)

        long_mask[exit_long_mask] = False
        short_mask[exit_short_mask] = False

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        df.loc[long_entry_mask, "bb_stop_long"] = close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        df.loc[long_entry_mask, "bb_tp_long"] = close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]
        df.loc[short_entry_mask, "bb_stop_short"] = close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        df.loc[short_entry_mask, "bb_tp_short"] = close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
