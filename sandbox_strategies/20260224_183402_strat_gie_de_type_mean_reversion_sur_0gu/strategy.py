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
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])

        # Prepare vortex conditions with lookback
        prev_vi_plus = np.roll(indicators['vortex']['vi_plus'], 1)
        prev_vi_minus = np.roll(indicators['vortex']['vi_minus'], 1)
        prev_vi_plus[0] = np.nan
        prev_vi_minus[0] = np.nan

        prev2_vi_plus = np.roll(indicators['vortex']['vi_plus'], 2)
        prev2_vi_minus = np.roll(indicators['vortex']['vi_minus'], 2)
        prev2_vi_plus[0] = prev2_vi_plus[1] = np.nan
        prev2_vi_minus[0] = prev2_vi_minus[1] = np.nan

        prev3_vi_plus = np.roll(indicators['vortex']['vi_plus'], 3)
        prev3_vi_minus = np.roll(indicators['vortex']['vi_minus'], 3)
        prev3_vi_plus[0] = prev3_vi_plus[1] = prev3_vi_plus[2] = np.nan
        prev3_vi_minus[0] = prev3_vi_minus[1] = prev3_vi_minus[2] = np.nan

        # Bollinger bands
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])

        # EMA trend
        ema_prev = np.roll(ema, 1)
        ema_prev[0] = np.nan

        # Entry conditions
        # Long entry: price touches lower band, vortex turns positive after 3 consecutive negative readings, EMA confirms uptrend
        long_entry_condition = (
            (close == indicators['bollinger']['lower']) &
            (indicators['vortex']['vi_plus'] > 0) &
            (prev_vi_plus < 0) &
            (prev2_vi_plus < 0) &
            (prev3_vi_plus < 0) &
            (ema > ema_prev)
        )

        # Short entry: price touches upper band, vortex turns negative after 3 consecutive positive readings, EMA confirms downtrend
        short_entry_condition = (
            (close == indicators['bollinger']['upper']) &
            (indicators['vortex']['vi_minus'] < 0) &
            (prev_vi_minus > 0) &
            (prev2_vi_minus > 0) &
            (prev3_vi_minus > 0) &
            (ema < ema_prev)
        )

        long_mask = long_entry_condition
        short_mask = short_entry_condition

        # Exit conditions
        # Exit long: price crosses upper band or vortex turns negative
        exit_long_condition = (
            (close > indicators['bollinger']['upper']) |
            (indicators['vortex']['vi_plus'] < 0)
        )

        # Exit short: price crosses lower band or vortex turns positive
        exit_short_condition = (
            (close < indicators['bollinger']['lower']) |
            (indicators['vortex']['vi_minus'] > 0)
        )

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Long entries
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        # Short entries
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
