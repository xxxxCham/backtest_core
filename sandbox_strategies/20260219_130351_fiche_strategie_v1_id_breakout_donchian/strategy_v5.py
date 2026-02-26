from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='donchian_vortex_atr_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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

        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        # Donchian bands
        dc = indicators['donchian']
        upper = np.nan_to_num(dc["upper"])
        middle = np.nan_to_num(dc["middle"])
        lower = np.nan_to_num(dc["lower"])

        # Vortex
        vx = indicators['vortex']
        vip = np.nan_to_num(vx["vi_plus"])
        vim = np.nan_to_num(vx["vi_minus"])

        # Helper cross functions
        prev_close = np.roll(close, 1); prev_close[0] = np.nan
        prev_upper = np.roll(upper, 1); prev_upper[0] = np.nan
        prev_lower = np.roll(lower, 1); prev_lower[0] = np.nan
        prev_middle = np.roll(middle, 1); prev_middle[0] = np.nan
        prev_vip = np.roll(vip, 1); prev_vip[0] = np.nan
        prev_vim = np.roll(vim, 1); prev_vim[0] = np.nan

        cross_up_close_upper = (close > upper) & (prev_close <= prev_upper)
        cross_down_close_lower = (close < lower) & (prev_close >= prev_lower)

        # Entry conditions
        long_mask = cross_up_close_upper & (vip > vim)
        short_mask = cross_down_close_lower & (vip < vim)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        cross_up_close_middle = (close > middle) & (prev_close <= prev_middle)
        cross_down_close_middle = (close < middle) & (prev_close >= prev_middle)
        cross_any_close_middle = cross_up_close_middle | cross_down_close_middle

        cross_down_vip_vim = (vip < vim) & (prev_vip >= prev_vim)
        cross_up_vip_vim = (vip > vim) & (prev_vip <= prev_vim)

        exit_mask_long = cross_any_close_middle | cross_down_vip_vim
        exit_mask_short = cross_any_close_middle | cross_up_vip_vim
        signals[exit_mask_long] = 0.0
        signals[exit_mask_short] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]
        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
