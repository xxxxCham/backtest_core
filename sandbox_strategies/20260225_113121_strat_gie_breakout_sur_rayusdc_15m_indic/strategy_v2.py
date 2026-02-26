from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rayusdc_ema_aroon_volume_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'aroon', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_sma_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
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
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
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
        ema = np.nan_to_num(indicators['ema'])
        aroon = indicators['aroon']
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # compute sma of volume oscillator
        volume_sma_period = int(params.get("volume_sma_period", 20))
        volume_sma = np.convolve(volume_osc, np.ones(volume_sma_period) / volume_sma_period, mode='valid')
        volume_sma = np.pad(volume_sma, (volume_sma_period - 1, 0), constant_values=np.nan)
        # detect crossovers
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        ema_crossover = (ema > prev_ema) & (prev_ema <= ema)
        ema_crossunder = (ema < prev_ema) & (prev_ema >= ema)
        # detect volume condition
        volume_condition = volume_osc > volume_sma
        # long entry condition
        long_condition = (ema_crossover) & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) & (volume_condition)
        long_mask = long_condition
        # short entry condition
        short_condition = (ema_crossunder) & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (volume_condition)
        short_mask = short_condition
        # exit condition (aroon momentum change)
        prev_aroon_up = np.roll(indicators['aroon']['aroon_up'], 1)
        prev_aroon_up[0] = np.nan
        prev_aroon_down = np.roll(indicators['aroon']['aroon_down'], 1)
        prev_aroon_down[0] = np.nan
        aroon_exit_long = (indicators['aroon']['aroon_up'] < indicators['aroon']['aroon_down']) & (prev_aroon_up >= prev_aroon_down)
        aroon_exit_short = (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up']) & (prev_aroon_down >= prev_aroon_up)
        # apply exits
        exit_long_mask = aroon_exit_long
        exit_short_mask = aroon_exit_short
        # handle signal overrides
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # apply ATR-based SL/TP
        stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))
        close = df["close"].values
        # init SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # compute SL/TP levels only for entry bars
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
