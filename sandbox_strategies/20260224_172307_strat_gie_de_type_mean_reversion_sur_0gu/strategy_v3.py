from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='williams_r_keltner_vortex_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'keltner', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 1.2,
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
            'keltner_period': ParameterSpec(
                name='keltner_period',
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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.2,
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
        williams_r = np.nan_to_num(indicators['williams_r'])
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['middle'] = np.nan_to_num(kelt["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        atr = np.nan_to_num(indicators['atr'])
        # Channel expansion check
        kelt_channel_width = indicators['keltner']['upper'] - indicators['keltner']['lower']
        prev_kelt_width = np.roll(kelt_channel_width, 1)
        prev_kelt_width[0] = np.nan
        channel_expanding = kelt_channel_width > prev_kelt_width
        # Cross helpers
        prev_williams_r = np.roll(williams_r, 1)
        prev_williams_r[0] = np.nan
        cross_below_neg80 = (williams_r < -80) & (prev_williams_r >= -80)
        cross_above_neg20 = (williams_r > -20) & (prev_williams_r <= -20)
        # Entry conditions
        long_entry = cross_below_neg80 & channel_expanding & (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (indicators['vortex']['vi_plus'] > 0.5)
        short_entry = cross_above_neg20 & channel_expanding & (indicators['vortex']['vi_plus'] < indicators['vortex']['vi_minus']) & (indicators['vortex']['vi_minus'] > 0.5)
        long_mask[long_entry] = True
        short_mask[short_entry] = True
        # Exit conditions
        exit_long = cross_above_neg20 | (williams_r < -80) | (df["close"] > indicators['keltner']['upper'])
        exit_short = cross_below_neg80 | (williams_r > -20) | (df["close"] < indicators['keltner']['lower'])
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # SL/TP setup
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Long SL/TP
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = df.loc[entry_long, "close"] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = df.loc[entry_long, "close"] + params["tp_atr_mult"] * atr[entry_long]
        # Short SL/TP
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = df.loc[entry_short, "close"] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = df.loc[entry_short, "close"] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
