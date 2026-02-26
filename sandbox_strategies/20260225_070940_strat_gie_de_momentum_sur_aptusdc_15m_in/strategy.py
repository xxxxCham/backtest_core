from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='roc_macd_adx_atr_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['roc', 'macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'atr_period': 14,
         'atr_vol_threshold': 0.0005,
         'exit_roc_threshold': 0.2,
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
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
                param_type='float',
                step=0.1,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
                default=50,
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
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # extract indicators with NaN handling
        roc = np.nan_to_num(indicators['roc'])
        macd_dict = indicators['macd']
        indicators['macd']['macd'] = np.nan_to_num(macd_dict["macd"])
        indicators['macd']['signal'] = np.nan_to_num(macd_dict["signal"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # parameters
        roc_thresh = params.get("roc_threshold", 0.5)
        exit_roc_thresh = params.get("exit_roc_threshold", 0.2)
        adx_thresh = params.get("adx_threshold", 25)
        atr_vol_thresh = params.get("atr_vol_threshold", 0.0005)
        stop_mult = params.get("stop_atr_mult", 1.3)
        tp_mult = params.get("tp_atr_mult", 2.21)

        # MACD cross detection
        prev_macd = np.roll(indicators['macd']['macd'], 1)
        prev_signal = np.roll(indicators['macd']['signal'], 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        macd_cross_up = (indicators['macd']['macd'] > indicators['macd']['signal']) & (prev_macd <= prev_signal)
        macd_cross_down = (indicators['macd']['macd'] < indicators['macd']['signal']) & (prev_macd >= prev_signal)

        # entry conditions
        long_entry = (roc > roc_thresh) & macd_cross_up & (adx_val > adx_thresh) & (atr > atr_vol_thresh)
        short_entry = (roc < -roc_thresh) & macd_cross_down & (adx_val > adx_thresh) & (atr > atr_vol_thresh)

        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # exit conditions
        exit_long = (np.abs(roc) < exit_roc_thresh) & macd_cross_down
        exit_short = (np.abs(roc) < exit_roc_thresh) & macd_cross_up

        # clear signals on exit
        long_mask[exit_long] = False
        short_mask[exit_short] = False

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR‑based stop‑loss / take‑profit columns
        close = df["close"].values
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
