from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_mean_reversion_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
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
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=8.0,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        bb = indicators['bollinger']
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        # prepare previous vortex values
        prev_vi_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vi_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_vi_plus[0] = np.nan
        prev_vi_minus[0] = np.nan
        # prepare 2nd, 3rd, 4th previous vortex values
        prev2_vi_plus = np.roll(indicators['vortex']['vi_plus'], 2)
        prev2_vi_minus = np.roll(indicators['vortex']['vi_minus'], 2)
        prev2_vi_plus[0] = np.nan
        prev2_vi_minus[0] = np.nan
        prev3_vi_plus = np.roll(indicators['vortex']['vi_plus'], 3)
        prev3_vi_minus = np.roll(indicators['vortex']['vi_minus'], 3)
        prev3_vi_plus[0] = np.nan
        prev3_vi_minus[0] = np.nan
        prev4_vi_plus = np.roll(indicators['vortex']['vi_plus'], 4)
        prev4_vi_minus = np.roll(indicators['vortex']['vi_minus'], 4)
        prev4_vi_plus[0] = np.nan
        prev4_vi_minus[0] = np.nan
        # long entry conditions
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        touch_lower_bb = close == lower_bb
        vortex_positive = indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']
        vortex_turned_positive = (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (prev_vi_plus <= prev_vi_minus)
        three_negatives = (prev_vi_plus <= prev_vi_minus) & (prev2_vi_plus <= prev2_vi_minus) & (prev3_vi_plus <= prev3_vi_minus)
        ema_upward = ema > np.roll(ema, 1)
        long_entry = touch_lower_bb & vortex_turned_positive & three_negatives & ema_upward
        long_mask = long_entry
        # short entry conditions
        touch_upper_bb = close == upper_bb
        vortex_negative = indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus']
        vortex_turned_negative = (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus']) & (prev_vi_plus >= prev_vi_minus)
        three_positives = (prev_vi_plus >= prev_vi_minus) & (prev2_vi_plus >= prev2_vi_minus) & (prev3_vi_plus >= prev3_vi_minus)
        ema_downward = ema < np.roll(ema, 1)
        short_entry = touch_upper_bb & vortex_turned_negative & three_positives & ema_downward
        short_mask = short_entry
        # exit conditions
        exit_long = close > upper_bb
        exit_short = close < lower_bb
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP levels for long entries
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        # set SL/TP levels for short entries
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
