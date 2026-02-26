from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='roc_macd_momentum_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'exit_roc_threshold': 0.2,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'roc_period': 9,
         'roc_threshold': 0.5,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.21,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'roc_period': ParameterSpec(
                name='roc_period',
                min_val=5,
                max_val=30,
                default=9,
                param_type='int',
                step=1,
            ),
            'roc_threshold': ParameterSpec(
                name='roc_threshold',
                min_val=0.1,
                max_val=5.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'exit_roc_threshold': ParameterSpec(
                name='exit_roc_threshold',
                min_val=0.0,
                max_val=5.0,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'macd_fast_period': ParameterSpec(
                name='macd_fast_period',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow_period': ParameterSpec(
                name='macd_slow_period',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=20,
                default=9,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=5.0,
                default=2.21,
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
        # initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # warmup protection
        signals.iloc[:warmup] = 0.0

        # extract indicator arrays
        roc = np.nan_to_num(indicators['roc'])
        macd_dict = indicators['macd']
        indicators['macd']['macd'] = np.nan_to_num(macd_dict["macd"])
        indicators['macd']['signal'] = np.nan_to_num(macd_dict["signal"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # parameters
        roc_thr = float(params.get("roc_threshold", 0.5))
        exit_roc_thr = float(params.get("exit_roc_threshold", 0.2))
        stop_mult = float(params.get("stop_atr_mult", 1.3))
        tp_mult = float(params.get("tp_atr_mult", 2.21))

        # MACD cross helpers
        prev_macd = np.roll(indicators['macd']['macd'], 1)
        prev_signal = np.roll(indicators['macd']['signal'], 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        macd_cross_up = (indicators['macd']['macd'] > indicators['macd']['signal']) & (prev_macd <= prev_signal)
        macd_cross_down = (indicators['macd']['macd'] < indicators['macd']['signal']) & (prev_macd >= prev_signal)

        # entry conditions
        long_mask = (roc > roc_thr) & macd_cross_up
        short_mask = (roc < -roc_thr) & macd_cross_down

        # exit conditions (implicitly flat because default is 0)
        # we do not need explicit masks; signals stay 0 when not entry

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # write ATR‑based stop‑loss and take‑profit on entry bars
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
