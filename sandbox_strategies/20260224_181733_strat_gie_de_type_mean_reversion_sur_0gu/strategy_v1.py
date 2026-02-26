from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='williams_r_vortex_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'vortex', 'atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.0,
         'vortex_period': 14,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=30,
                default=14,
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
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
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

        # Extract indicators
        williams_r = np.nan_to_num(indicators['williams_r'])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        atr = np.nan_to_num(indicators['atr'])
        ema_20 = np.nan_to_num(indicators['ema'])

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        # Previous ATR
        prev_atr = np.roll(atr, 1)
        prev_atr[0] = np.nan

        # Williams R crosses below -80 (overbought)
        prev_williams_r = np.roll(williams_r, 1)
        prev_williams_r[0] = np.nan
        williams_cross_down = (williams_r < -80) & (prev_williams_r >= -80)

        # ATR increasing
        atr_increasing = (atr > prev_atr) & (prev_atr > 0)

        # Vortex confirmation
        vortex_bullish = (indicators['vortex']['vi_plus'] > 0.5) & (indicators['vortex']['vi_minus'] < 0.5)
        vortex_bearish = (indicators['vortex']['vi_minus'] > 0.5) & (indicators['vortex']['vi_plus'] < 0.5)

        # Long entry
        long_entry = williams_cross_down & atr_increasing & vortex_bullish
        long_mask = long_entry

        # Short entry
        williams_cross_up = (williams_r > -80) & (prev_williams_r <= -80)
        short_entry = williams_cross_up & atr_increasing & vortex_bearish
        short_mask = short_entry

        # Exit conditions
        # Close crosses below EMA(20)
        prev_close = np.roll(df["close"].values, 1)
        prev_close[0] = np.nan
        close_ema_cross_down = (df["close"].values < ema_20) & (prev_close >= ema_20)

        # Vortex signal for exit
        vortex_exit_long = (indicators['vortex']['vi_plus'] > 0.5) & (indicators['vortex']['vi_minus'] < 0.5)
        vortex_exit_short = (indicators['vortex']['vi_minus'] > 0.5) & (indicators['vortex']['vi_plus'] < 0.5)

        # Combine exit conditions
        exit_long = close_ema_cross_down | vortex_exit_long
        exit_short = close_ema_cross_down | vortex_exit_short

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exits
        exit_long_mask = signals == 1.0
        exit_short_mask = signals == -1.0

        # Update exit conditions
        signals[exit_long_mask & (close_ema_cross_down | vortex_exit_long)] = 0.0
        signals[exit_short_mask & (close_ema_cross_down | vortex_exit_short)] = 0.0

        # Risk management
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        tp_atr_mult = params.get("tp_atr_mult", 2.0)

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        entry_long_mask = signals == 1.0
        if entry_long_mask.any():
            close = df["close"].values
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        # Short entry SL/TP
        entry_short_mask = signals == -1.0
        if entry_short_mask.any():
            close = df["close"].values
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals