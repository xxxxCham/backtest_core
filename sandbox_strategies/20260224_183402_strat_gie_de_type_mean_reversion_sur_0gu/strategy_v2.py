from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_vortex_ema')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'vortex', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'ema_period': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
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
        bb = indicators['bollinger']
        upper_bb = np.nan_to_num(bb["upper"])
        middle_bb = np.nan_to_num(bb["middle"])
        lower_bb = np.nan_to_num(bb["lower"])
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        vortex_osc = indicators['vortex']['vi_plus'] - indicators['vortex']['vi_minus']
        # Precompute previous vortex values for trend confirmation
        prev_vortex = np.roll(vortex_osc, 1)
        prev_vortex[0] = np.nan
        prev2_vortex = np.roll(vortex_osc, 2)
        prev2_vortex[0] = prev2_vortex[1] = np.nan
        prev3_vortex = np.roll(vortex_osc, 3)
        prev3_vortex[0] = prev3_vortex[1] = prev3_vortex[2] = np.nan
        # Precompute EMA values for trend confirmation
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        # Entry conditions
        # Long entry: price touches lower Bollinger band AND VORTEX turns positive after 3 negative readings AND EMA trend confirms
        price_touches_lower = (df["close"].values == lower_bb)
        vortex_turns_positive = (vortex_osc > 0) & (prev_vortex < 0) & (prev2_vortex < 0) & (prev3_vortex < 0)
        ema_trend_confirms = (ema > prev_ema)
        long_entry = price_touches_lower & vortex_turns_positive & ema_trend_confirms
        long_mask = long_entry
        # Short entry: price touches upper Bollinger band AND VORTEX turns negative after 3 positive readings AND EMA trend confirms
        price_touches_upper = (df["close"].values == upper_bb)
        vortex_turns_negative = (vortex_osc < 0) & (prev_vortex > 0) & (prev2_vortex > 0) & (prev3_vortex > 0)
        ema_trend_confirms_short = (ema < prev_ema)
        short_entry = price_touches_upper & vortex_turns_negative & ema_trend_confirms_short
        short_mask = short_entry
        # Exit conditions
        # Exit long: price crosses above upper Bollinger band OR VORTEX turns negative again
        price_crosses_upper = (df["close"].values > upper_bb)
        vortex_turns_negative_again = (vortex_osc < 0)
        exit_long = price_crosses_upper | vortex_turns_negative_again
        # Exit short: price crosses below lower Bollinger band OR VORTEX turns positive again
        price_crosses_lower = (df["close"].values < lower_bb)
        vortex_turns_positive_again = (vortex_osc > 0)
        exit_short = price_crosses_lower | vortex_turns_positive_again
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Risk management
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        # Initialize SL/TP columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Compute ATR-based SL/TP on entry bars only
        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)
        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = df.loc[entry_long_mask, "close"] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = df.loc[entry_long_mask, "close"] + tp_atr_mult * atr[entry_long_mask]
        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = df.loc[entry_short_mask, "close"] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = df.loc[entry_short_mask, "close"] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
