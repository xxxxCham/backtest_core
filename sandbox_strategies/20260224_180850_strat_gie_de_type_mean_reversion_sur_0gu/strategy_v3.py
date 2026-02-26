from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_keltner_stochastic_volume_adjusted')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'keltner', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stoch_d_period': 3,
         'stoch_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 1.5,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_k_period': ParameterSpec(
                name='stoch_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'volume_oscillator_short': ParameterSpec(
                name='volume_oscillator_short',
                min_val=5,
                max_val=20,
                default=12,
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
                min_val=1.0,
                max_val=4.5,
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
        stoch = indicators['stochastic']
        kelt = indicators['keltner']
        vol = indicators['volume_oscillator']
        atr = np.nan_to_num(indicators['atr'])

        # Stochastic values
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])

        # Keltner values
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])

        # Volume oscillator
        vol_osc = np.nan_to_num(vol)

        # Previous values for crossovers
        prev_stoch_k = np.roll(indicators['stochastic']['stoch_k'], 1)
        prev_stoch_k[0] = np.nan
        prev_kelt_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_kelt_lower[0] = np.nan
        prev_kelt_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_kelt_upper[0] = np.nan

        # Crossover conditions
        cross_below_20 = (indicators['stochastic']['stoch_k'] < 20) & (prev_stoch_k >= 20)
        cross_above_80 = (indicators['stochastic']['stoch_k'] > 80) & (prev_stoch_k <= 80)

        # Keltner channel contraction
        keltner_contracting_long = indicators['keltner']['lower'] > prev_kelt_lower
        keltner_contracting_short = indicators['keltner']['upper'] < prev_kelt_upper

        # Volume filter
        vol_filter_long = vol_osc > 1.5
        vol_filter_short = vol_osc > 1.5

        # Entry conditions
        long_entry = cross_below_20 & keltner_contracting_long & vol_filter_long
        short_entry = cross_above_80 & keltner_contracting_short & vol_filter_short

        # Set long and short masks
        long_mask = long_entry
        short_mask = short_entry

        # Signal assignment
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        close = df["close"].values
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set SL/TP levels on entry bars only
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)

        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
