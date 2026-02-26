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
        # compute volume sma
        volume_sma_period = int(params.get("volume_sma_period", 20))
        volume_sma = np.convolve(volume_osc, np.ones(volume_sma_period), mode='valid') / volume_sma_period
        volume_sma = np.pad(volume_sma, (volume_sma_period - 1, 0), constant_values=np.nan)
        # prepare crossover logic
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        ema_cross_up = (ema > prev_ema) & (np.roll(ema, 1) <= prev_ema)
        ema_cross_down = (ema < prev_ema) & (np.roll(ema, 1) >= prev_ema)
        # entry conditions
        long_entry = ema_cross_up & (indicators['aroon']['aroon_up'] > indicators['aroon']['aroon_down']) & (volume_osc > volume_sma)
        short_entry = ema_cross_down & (indicators['aroon']['aroon_down'] > indicators['aroon']['aroon_up']) & (volume_osc > volume_sma)
        # exit conditions
        aroon_up_cross_down = (indicators['aroon']['aroon_up'] < indicators['aroon']['aroon_down']) & (np.roll(indicators['aroon']['aroon_up'], 1) >= np.roll(indicators['aroon']['aroon_down'], 1))
        aroon_down_cross_up = (indicators['aroon']['aroon_down'] < indicators['aroon']['aroon_up']) & (np.roll(indicators['aroon']['aroon_down'], 1) >= np.roll(indicators['aroon']['aroon_up'], 1))
        exit_long = aroon_up_cross_down
        exit_short = aroon_down_cross_up
        # apply signals
        long_mask = long_entry
        short_mask = short_entry
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # exit signals
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0
        # set stop-loss and take-profit
        close = df["close"].values
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
