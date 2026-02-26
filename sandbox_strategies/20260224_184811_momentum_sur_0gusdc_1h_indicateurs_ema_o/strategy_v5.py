from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_vortex_ema_obv')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'vortex', 'atr', 'rsi', 'momentum']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 50,
         'leverage': 1,
         'obv_period': 20,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'obv_period': ParameterSpec(
                name='obv_period',
                min_val=5,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        vortex = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        rsi = np.nan_to_num(indicators['rsi'])
        momentum = np.nan_to_num(indicators['momentum'])
        # prepare previous values for crossovers
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan
        prev_vortex = np.roll(vortex, 1)
        prev_vortex[0] = np.nan
        prev_momentum = np.roll(momentum, 1)
        prev_momentum[0] = np.nan
        prev2_momentum = np.roll(momentum, 2)
        prev2_momentum[0] = prev2_momentum[1] = np.nan
        # long entry conditions
        long_condition1 = close > ema
        long_condition2 = obv > prev_obv
        long_condition3 = vortex > 0.5
        long_condition4 = vortex > prev_vortex
        long_entry = long_condition1 & long_condition2 & long_condition3 & long_condition4
        long_mask = long_entry
        # short entry conditions
        short_condition1 = close < ema
        short_condition2 = obv < prev_obv
        short_condition3 = vortex > 0.5
        short_condition4 = vortex > prev_vortex
        short_entry = short_condition1 & short_condition2 & short_condition3 & short_condition4
        short_mask = short_entry
        # exit conditions
        exit_rsi = rsi > params["rsi_overbought"]
        exit_momentum = (momentum < 0) & (prev_momentum < 0) & (prev2_momentum < 0)
        exit_mask = exit_rsi | exit_momentum
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # long SL/TP
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # short SL/TP
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
