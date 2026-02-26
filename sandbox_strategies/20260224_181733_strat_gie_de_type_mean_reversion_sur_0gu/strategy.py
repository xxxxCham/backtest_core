from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='williams_r_vortex_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'vortex', 'atr', 'ema']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.0,
         'warmup': 50,
         'williams_r_overbought': 80,
         'williams_r_oversold': 20,
         'williams_r_period': 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_r_period': ParameterSpec(
                name='williams_r_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
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
                default=2.0,
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
        williams_r = np.nan_to_num(indicators['williams_r'])
        vortex = indicators['vortex']
        atr = np.nan_to_num(indicators['atr'])
        ema_20 = np.nan_to_num(indicators['ema'])

        # Compute previous atr for comparison
        prev_atr = np.roll(atr, 1)
        prev_atr[0] = np.nan

        # Define entry conditions
        # Long entry: williams_r < -80 AND atr > np.roll(atr, 2) AND vortex < 0.5
        long_entry_condition = (williams_r < -params["williams_r_overbought"]) & (atr > prev_atr) & (indicators['vortex']["oscillator"] < 0.5)

        # Short entry: williams_r > -20 AND atr > np.roll(atr, 2) AND vortex < 0.5
        short_entry_condition = (williams_r > -params["williams_r_oversold"]) & (atr > prev_atr) & (indicators['vortex']["oscillator"] < 0.5)

        # Define exit conditions
        # Exit on EMA(20) cross below
        close = df["close"].values
        ema_20 = np.nan_to_num(indicators['ema'])
        prev_ema_20 = np.roll(ema_20, 1)
        prev_ema_20[0] = np.nan
        exit_long_condition = close < ema_20
        exit_short_condition = close > ema_20

        # Exit on vortex oscillator crossing above 0.5
        exit_long_condition |= (indicators['vortex']["oscillator"] > 0.5)
        exit_short_condition |= (indicators['vortex']["oscillator"] > 0.5)

        # Compute long and short masks
        long_mask = long_entry_condition
        short_mask = short_entry_condition

        # Apply exit conditions to active positions
        # For simplicity, assume no overlapping positions
        # Exit longs
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set warmup period
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        if entry_long_mask.any():
            df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
            df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        if entry_short_mask.any():
            df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
            df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals