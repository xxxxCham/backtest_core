from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_inverse_mode')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'supertrend', 'atr', 'ema', 'pivot_points']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.0,
         'supertrend_period': 10,
         'tp_atr_mult': 2.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'bollinger_period': ParameterSpec(
                name='bollinger_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=20,
                default=10,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.0,
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
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        vt = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vt["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vt["vi_minus"])
        signal_vt = np.nan_to_num(vt["signal"])
        st = indicators['supertrend']
        direction_st = np.nan_to_num(st["direction"])
        atr = np.nan_to_num(indicators['atr'])
        ema = np.nan_to_num(indicators['ema'])
        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        close = df["close"].values
        # Cross helpers
        prev_vi_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vi_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_vi_plus[0] = np.nan
        prev_vi_minus[0] = np.nan
        cross_down_vortex = (indicators['vortex']['vi_plus'] < signal_vt) & (prev_vi_plus >= signal_vt)
        cross_up_vortex = (indicators['vortex']['vi_plus'] > signal_vt) & (prev_vi_plus <= signal_vt)
        # Touch upper/lower band
        close_upper = (close == upper_bb)
        close_lower = (close == lower_bb)
        # Supertrend regime is flat (0)
        regime_flat = (direction_st == 0)
        # Long entry conditions
        long_entry = close_upper & cross_down_vortex & regime_flat
        long_mask[long_entry] = True
        # Short entry conditions
        short_entry = close_lower & cross_up_vortex & regime_flat
        short_mask[short_entry] = True
        # Exit conditions
        ema_cross_down = (close < ema) & (np.roll(close, 1) >= np.roll(ema, 1))
        ema_cross_down[0] = False
        r1_cross_up = (close > r1) & (np.roll(close, 1) <= np.roll(r1, 1))
        r1_cross_up[0] = False
        # Apply exits
        exit_long_mask = long_mask & (ema_cross_down | r1_cross_up)
        exit_short_mask = short_mask & (ema_cross_down | r1_cross_up)
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # Set entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
