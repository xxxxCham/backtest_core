from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='style_breakout_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_200_period': 200,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=30,
                default=20,
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
                min_val=1.5,
                max_val=5.0,
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
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        ema_200 = np.nan_to_num(indicators['ema'])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        # compute previous values for crossovers
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan
        prev_ema_200 = np.roll(ema_200, 1)
        prev_ema_200[0] = np.nan
        prev_volume_osc = np.roll(volume_osc, 1)
        prev_volume_osc[0] = np.nan
        # long entry: close crosses above upper band, ema_200 trending up, volume_osc > 0
        long_entry = (close > upper) & (prev_upper <= upper) & (ema_200 > prev_ema_200) & (volume_osc > 0)
        long_mask = long_entry
        # short entry: close crosses below lower band, ema_200 trending down, volume_osc < 0
        short_entry = (close < lower) & (prev_lower >= lower) & (ema_200 < prev_ema_200) & (volume_osc < 0)
        short_mask = short_entry
        # exit conditions
        # exit long: close crosses below middle band or volume_osc < 0 or ema_200 turns down
        long_exit = (close < middle) & (prev_middle >= middle) | (volume_osc < 0) | (ema_200 < prev_ema_200)
        # exit short: close crosses above middle band or volume_osc > 0 or ema_200 turns up
        short_exit = (close > middle) & (prev_middle <= middle) | (volume_osc > 0) | (ema_200 > prev_ema_200)
        # apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # set SL/TP based on ATR
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = signals == 1.0
        entry_short = signals == -1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
