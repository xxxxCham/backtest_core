from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_stochastic_williams_volume')

    @property
    def required_indicators(self) -> List[str]:
        return ['stochastic', 'williams_r', 'volume_oscillator', 'atr', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_long': 26,
         'volume_oscillator_short': 12,
         'warmup': 50,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=30,
                default=14,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])
        williams_r = np.nan_to_num(indicators['williams_r'])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        # Compute previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        # Long entry conditions
        long_cross_below_20 = (k < 20) & (prev_k >= 20)
        long_williams_r_below_80 = williams_r < -80
        long_vol_osc_positive = vol_osc > 0
        vol_osc_sma = np.nan_to_num(np.convolve(vol_osc, np.ones(5)/5, mode='valid'))
        vol_osc_sma_padded = np.pad(vol_osc_sma, (4, 0), mode='constant', constant_values=np.nan)
        long_vol_osc_above_sma = vol_osc > vol_osc_sma_padded
        long_entry = long_cross_below_20 & long_williams_r_below_80 & long_vol_osc_positive & long_vol_osc_above_sma
        long_mask = long_entry
        # Short entry conditions
        short_cross_above_80 = (k > 80) & (prev_k <= 80)
        short_williams_r_above_neg20 = williams_r > -20
        short_vol_osc_negative = vol_osc < 0
        short_vol_osc_below_sma = vol_osc < vol_osc_sma_padded
        short_entry = short_cross_above_80 & short_williams_r_above_neg20 & short_vol_osc_negative & short_vol_osc_below_sma
        short_mask = short_entry
        # Exit conditions
        exit_long = (k > 80) | (df["close"] >= indicators['bollinger']['upper'])
        exit_short = (k < 20) | (df["close"] <= indicators['bollinger']['upper'])
        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set SL/TP levels for long entries
        entry_long_mask = (signals == 1.0)
        close = df["close"].values
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        # Set SL/TP levels for short entries
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
