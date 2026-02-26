from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_psar_atr_breakout_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'psar', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'psar_max_step': 0.2,
         'psar_step': 0.02,
         'stop_atr_mult': 1.8,
         'tp_atr_mult': 3.9,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_multiplier': ParameterSpec(
                name='keltner_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'psar_step': ParameterSpec(
                name='psar_step',
                min_val=0.01,
                max_val=0.05,
                default=0.02,
                param_type='float',
                step=0.1,
            ),
            'psar_max_step': ParameterSpec(
                name='psar_max_step',
                min_val=0.1,
                max_val=0.5,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
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
                default=1.8,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.9,
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

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        middle = np.nan_to_num(kelt["middle"])
        lower = np.nan_to_num(kelt["lower"])

        indicators['psar']['sar'] = np.nan_to_num(indicators['psar']["sar"])

        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan
        prev_sar = np.roll(indicators['psar']['sar'], 1)
        prev_sar[0] = np.nan

        # Long entry: close crosses above upper and PSAR below close
        cross_up_upper = (close > upper) & (prev_close <= prev_upper)
        long_mask = cross_up_upper & (indicators['psar']['sar'] < close)

        # Short entry: close crosses below lower and PSAR above close
        cross_down_lower = (close < lower) & (prev_close >= prev_lower)
        short_mask = cross_down_lower & (indicators['psar']['sar'] > close)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Long exit: close crosses below middle or PSAR flips above close
        cross_down_middle = (close < middle) & (prev_close >= prev_middle)
        cross_down_sar = (close < indicators['psar']['sar']) & (prev_close >= prev_sar)
        long_exit_mask = cross_down_middle | cross_down_sar

        # Short exit: close crosses above middle or PSAR flips below close
        cross_up_middle = (close > middle) & (prev_close <= prev_middle)
        cross_up_sar = (close > indicators['psar']['sar']) & (prev_close <= prev_sar)
        short_exit_mask = cross_up_middle | cross_up_sar

        # Apply exits: set signals to 0.0 on exit bars
        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # ATR-based SL/TP for long positions
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.8)
        tp_atr_mult = params.get("tp_atr_mult", 3.9)

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
