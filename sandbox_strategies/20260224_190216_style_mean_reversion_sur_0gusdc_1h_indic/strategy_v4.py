from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_rsi_vorticity_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['stoch_rsi', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stoch_rsi_overbought': 80,
         'stoch_rsi_oversold': 20,
         'stoch_rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 1.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_rsi_period': ParameterSpec(
                name='stoch_rsi_period',
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
                default=1.5,
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
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # Extract indicators
        stoch_rsi = indicators['stoch_rsi']
        k = np.nan_to_num(indicators['stoch_rsi']["k"])
        d = np.nan_to_num(indicators['stoch_rsi']["d"])
        vortex = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        # Previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_vortex = np.roll(vortex, 1)
        prev_atr = np.roll(atr, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        prev_vortex[0] = np.nan
        prev_atr[0] = np.nan
        # Entry conditions
        # Long entry: k crosses below oversold, vortex < 0, atr increasing
        long_entry = (k < params["stoch_rsi_oversold"]) & (prev_k >= params["stoch_rsi_oversold"]) & (vortex < 0) & (prev_vortex >= 0) & (atr > prev_atr)
        long_mask = long_entry
        # Short entry: k crosses above overbought, vortex > 0, atr increasing
        short_entry = (k > params["stoch_rsi_overbought"]) & (prev_k <= params["stoch_rsi_overbought"]) & (vortex > 0) & (prev_vortex <= 0) & (atr > prev_atr)
        short_mask = short_entry
        # Exit conditions
        # Exit long: k crosses above overbought OR vortex > 0
        long_exit = (k > params["stoch_rsi_overbought"]) & (prev_k <= params["stoch_rsi_overbought"]) | (vortex > 0) & (prev_vortex <= 0)
        # Exit short: k crosses below oversold OR vortex < 0
        short_exit = (k < params["stoch_rsi_oversold"]) & (prev_k >= params["stoch_rsi_oversold"]) | (vortex < 0) & (prev_vortex >= 0)
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set SL/TP levels for long entries
        entry_mask_long = (signals == 1.0)
        if entry_mask_long.any():
            close = df["close"].values
            df.loc[:, "bb_stop_long"] = np.nan
            df.loc[:, "bb_tp_long"] = np.nan
            df.loc[entry_mask_long, "bb_stop_long"] = close[entry_mask_long] - params["stop_atr_mult"] * atr[entry_mask_long]
            df.loc[entry_mask_long, "bb_tp_long"] = close[entry_mask_long] + params["tp_atr_mult"] * atr[entry_mask_long]
        # Set SL/TP levels for short entries
        entry_mask_short = (signals == -1.0)
        if entry_mask_short.any():
            close = df["close"].values
            df.loc[:, "bb_stop_short"] = np.nan
            df.loc[:, "bb_tp_short"] = np.nan
            df.loc[entry_mask_short, "bb_stop_short"] = close[entry_mask_short] + params["stop_atr_mult"] * atr[entry_mask_short]
            df.loc[entry_mask_short, "bb_tp_short"] = close[entry_mask_short] - params["tp_atr_mult"] * atr[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals
