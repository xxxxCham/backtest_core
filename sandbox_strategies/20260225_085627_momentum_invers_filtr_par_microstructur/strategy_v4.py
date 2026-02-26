from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='filtered_momentum_reversal_european')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'onchain_smoothing', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'european_hours_end': 16,
         'european_hours_start': 8,
         'leverage': 1,
         'onchain_smoothing_period': 5,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 2.0,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'onchain_smoothing_period': ParameterSpec(
                name='onchain_smoothing_period',
                min_val=3,
                max_val=10,
                default=5,
                param_type='int',
                step=1,
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

        # Extract and clean indicators
        williams_r = np.nan_to_num(indicators['williams_r'])
        onchain_smoothing = np.nan_to_num(indicators['onchain_smoothing'])
        atr = np.nan_to_num(indicators['atr'])

        # Get hour from timestamp
        hours = df.index.hour.values

        # European hours mask
        euro_start = int(params.get("european_hours_start", 8))
        euro_end = int(params.get("european_hours_end", 16))
        euro_hours_mask = (hours >= euro_start) & (hours <= euro_end)

        # On-chain smoothing trend detection
        prev_onchain = np.roll(onchain_smoothing, 1)
        prev_onchain[0] = np.nan
        onchain_uptrend = onchain_smoothing > prev_onchain
        onchain_downtrend = onchain_smoothing < prev_onchain

        # Entry conditions
        long_entry = (williams_r > -20) & onchain_uptrend & euro_hours_mask
        short_entry = (williams_r < -80) & onchain_downtrend & euro_hours_mask

        # Apply entry signals
        long_mask = long_entry
        short_mask = short_entry

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Get ATR multiplier parameters
        stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

        # Get close prices
        close = df["close"].values

        # Set SL/TP levels for long positions
        long_entry_mask = signals == 1.0
        df.loc[long_entry_mask, "bb_stop_long"] = close[long_entry_mask] - stop_atr_mult * atr[long_entry_mask]
        df.loc[long_entry_mask, "bb_tp_long"] = close[long_entry_mask] + tp_atr_mult * atr[long_entry_mask]

        # Set SL/TP levels for short positions
        short_entry_mask = signals == -1.0
        df.loc[short_entry_mask, "bb_stop_short"] = close[short_entry_mask] + stop_atr_mult * atr[short_entry_mask]
        df.loc[short_entry_mask, "bb_tp_short"] = close[short_entry_mask] - tp_atr_mult * atr[short_entry_mask]
        signals.iloc[:warmup] = 0.0
        return signals
