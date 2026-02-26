from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='aptusdc_mfi_macd_atr_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_vol_threshold': 0.0005,
         'leverage': 1,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'mfi_period': 14,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=20,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal_period': ParameterSpec(
                name='macd_signal_period',
                min_val=5,
                max_val=15,
                default=9,
                param_type='int',
                step=1,
            ),
            'mfi_period': ParameterSpec(
                name='mfi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
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
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.5,
                max_val=6.0,
                default=3.6,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=5,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # indicators (sanitized)
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        mfi = np.nan_to_num(indicators['mfi'])
        atr = np.nan_to_num(indicators['atr'])

        # parameters
        atr_vol_threshold = float(params.get("atr_vol_threshold", 0.0005))
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.6))

        # entry conditions
        long_entry = (macd_hist > 0) & (mfi > 50) & (atr > atr_vol_threshold)
        short_entry = (macd_hist < 0) & (mfi < 50) & (atr > atr_vol_threshold)

        # cross detection for exits
        prev_macd = np.roll(macd_hist, 1)
        prev_macd[0] = np.nan
        macd_cross_down = (macd_hist < 0) & (prev_macd >= 0)   # + to -
        macd_cross_up = (macd_hist > 0) & (prev_macd <= 0)     # - to +

        prev_mfi = np.roll(mfi, 1)
        prev_mfi[0] = np.nan
        mfi_cross_down = (mfi < 50) & (prev_mfi >= 50)
        mfi_cross_up = (mfi > 50) & (prev_mfi <= 50)

        long_exit = macd_cross_down | mfi_cross_down
        short_exit = macd_cross_up | mfi_cross_up

        # final masks (prevent entry and immediate exit on same bar)
        long_mask = long_entry & ~long_exit
        short_mask = short_entry & ~short_exit

        # assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close_prices = df["close"].values

        # long ATR-based SL/TP
        if long_mask.any():
            entry_price_long = close_prices[long_mask]
            atr_long = atr[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_atr_mult * atr_long
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_atr_mult * atr_long

        # short ATR-based SL/TP
        if short_mask.any():
            entry_price_short = close_prices[short_mask]
            atr_short = atr[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_atr_mult * atr_short
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_atr_mult * atr_short

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
