from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='frontusdc_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
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
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])
        # compute EMA of volume oscillator
        ema_volume_oscillator = np.nan_to_num(np.roll(volume_oscillator, 1))
        ema_volume_oscillator[0] = volume_oscillator[0]
        # entry conditions
        # long entry: close touches lower EMA band AND volume_oscillator > EMA of volume_oscillator AND williams_r < -80
        close_to_lower_ema = np.abs(close - ema) < 0.001 * ema  # approximate touch
        volume_above_ema = volume_oscillator > ema_volume_oscillator
        williams_r_oversold = williams_r < -80
        long_entry = close_to_lower_ema & volume_above_ema & williams_r_oversold
        long_mask = long_entry
        # short entry: close touches upper EMA band AND volume_oscillator > EMA of volume_oscillator AND williams_r > -20
        close_to_upper_ema = np.abs(close - ema) < 0.001 * ema  # approximate touch
        williams_r_overbought = williams_r > -20
        short_entry = close_to_upper_ema & volume_above_ema & williams_r_overbought
        short_mask = short_entry
        # exit conditions
        # exit on 50-period EMA cross (use a simple EMA of 50 periods for exit)
        ema_50 = np.nan_to_num(indicators['ema'])
        prev_ema_50 = np.roll(ema_50, 1)
        prev_ema_50[0] = ema_50[0]
        ema_cross_down = (close < ema_50) & (prev_ema_50 >= ema_50)
        williams_r_exit = williams_r > -20
        exit_condition = ema_cross_down | williams_r_exit
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP levels
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
