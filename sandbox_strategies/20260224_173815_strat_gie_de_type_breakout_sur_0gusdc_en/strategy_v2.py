from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_volatility_filter_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_atr_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
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
            'keltner_atr_mult': ParameterSpec(
                name='keltner_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
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
                min_val=2.0,
                max_val=4.5,
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

        # Extract indicators
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])

        # ATR volatility filter
        atr_7d = np.nan_to_num(indicators['atr'])
        median_atr = np.nanmedian(atr_7d)
        vol_filter = atr < median_atr

        # Volume confirmation
        vol_confirm = vol_osc > 0

        # Entry conditions
        close = df["close"].values

        # Previous values for crossovers
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_upper[0] = np.nan
        prev_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_lower[0] = np.nan
        prev_middle = np.roll(indicators['keltner']['middle'], 1)
        prev_middle[0] = np.nan

        # Long entry: close crosses above indicators['keltner']['upper'] with volume > 0 and ATR < median
        long_entry = (close > indicators['keltner']['upper']) & (prev_close <= prev_upper) & vol_confirm & vol_filter
        long_mask = long_entry

        # Short entry: close crosses below indicators['keltner']['lower'] with volume > 0 and ATR < median
        short_entry = (close < indicators['keltner']['lower']) & (prev_close >= prev_lower) & vol_confirm & vol_filter
        short_mask = short_entry

        # Exit conditions
        # Exit long: close crosses below keltner middle
        long_exit = (close < indicators['keltner']['middle']) & (prev_close >= prev_middle)
        # Exit short: close crosses above keltner middle
        short_exit = (close > indicators['keltner']['middle']) & (prev_close <= prev_middle)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Apply exits
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Write SL/TP columns into df if using ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # On long entry bars only
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # On short entry bars only
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
