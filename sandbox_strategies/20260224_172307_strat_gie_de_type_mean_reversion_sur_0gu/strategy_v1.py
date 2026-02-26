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
        # warmup protection
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

        # Channel expansion detection
        prev_kelt_upper = np.roll(indicators['keltner']['upper'], 1)
        prev_kelt_lower = np.roll(indicators['keltner']['lower'], 1)
        prev_kelt_upper[0] = np.nan
        prev_kelt_lower[0] = np.nan
        channel_expanding = (indicators['keltner']['upper'] - indicators['keltner']['lower']) > (prev_kelt_upper - prev_kelt_lower)

        # Vortex explosion detection (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus'] and both > 0.5)
        vortex_explosion = (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (indicators['vortex']['vi_plus'] > 0.5) & (indicators['vortex']['vi_minus'] > 0.5)

        # Cross detection helpers
        prev_williams_r = np.roll(williams_r, 1)
        prev_williams_r[0] = np.nan
        cross_below_neg80 = (williams_r < -80) & (prev_williams_r >= -80)
        cross_above_neg20 = (williams_r > -20) & (prev_williams_r <= -20)

        # Long entry: Williams %R crosses below -80 with expanding Keltner channel and vortex explosion
        long_entry = cross_below_neg80 & channel_expanding & vortex_explosion

        # Short entry: Williams %R crosses above -20 with expanding Keltner channel and vortex explosion
        short_entry = cross_above_neg20 & channel_expanding & vortex_explosion

        # Exit conditions
        long_exit = cross_above_neg20 | (df["close"].values > indicators['keltner']['upper'])
        short_exit = cross_below_neg80 | (df["close"].values < indicators['keltner']['lower'])

        # Apply signals
        long_mask = long_entry
        short_mask = short_entry

        # Update signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management - ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entries
        entry_long = signals == 1.0
        if entry_long.any():
            close = df["close"].values
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            # TP at 1.2x distance to expected reversion level (Keltner middle)
            expected_rev = indicators['keltner']['middle'][entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * (expected_rev - close[entry_long])

        # Short entries
        entry_short = signals == -1.0
        if entry_short.any():
            close = df["close"].values
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            # TP at 1.2x distance to expected reversion level (Keltner middle)
            expected_rev = indicators['keltner']['middle'][entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * (close[entry_short] - expected_rev)
        signals.iloc[:warmup] = 0.0
        return signals
