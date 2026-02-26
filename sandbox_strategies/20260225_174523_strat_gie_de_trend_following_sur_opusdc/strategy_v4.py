from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aroon_vortex_atr_trend')

    @property
    def required_indicators(self) -> List[str]:
        return ['aroon', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 25,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.8,
         'tp_atr_mult': 1.9,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=50,
                default=25,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
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
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=1.9,
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

        # extract indicator arrays
        aroon = indicators['aroon']
        vortex = indicators['vortex']
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        indicators['aroon']['aroon_up'] = indicators['aroon']["aroon_up"]
        indicators['aroon']['aroon_down'] = indicators['aroon']["aroon_down"]
        indicators['vortex']['vi_plus'] = indicators['vortex']["vi_plus"]
        indicators['vortex']['vi_minus'] = indicators['vortex']["vi_minus"]

        # helper cross functions
        prev_up = np.roll(indicators['aroon']['aroon_up'], 1)
        prev_down = np.roll(indicators['aroon']['aroon_down'], 1)
        prev_up[0] = np.nan
        prev_down[0] = np.nan
        cross_up = (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) & (prev_up <= prev_down)
        cross_down = (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (prev_down <= prev_up)

        # long entry conditions
        long_cond = (
            cross_up
            & (indicators['aroon']['aroon_up'] > 70.0)
            & (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus'])
        )
        long_mask = long_cond

        # short entry conditions
        short_cond = (
            cross_down
            & (indicators['aroon']['aroon_down'] > 70.0)
            & (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus'])
        )
        short_mask = short_cond

        # exit conditions (trend reversal)
        exit_long = (
            (indicators['aroon']['aroon_up'] < indicators['aroon']['aroon_down'])
            | (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus'])
        )
        exit_short = (
            (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up'])
            | (indicators['vortex']['vi_minus'] < indicators['vortex']['vi_plus'])
        )

        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # flatten on reversal
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # write ATR based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.8)
        tp_atr_mult = params.get("tp_atr_mult", 1.9)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
