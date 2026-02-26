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
        # Extract indicators
        stoch_rsi = indicators['stoch_rsi']
        k = np.nan_to_num(indicators['stoch_rsi']["k"])
        d = np.nan_to_num(indicators['stoch_rsi']["d"])
        vortex = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(indicators['vortex']["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(indicators['vortex']["vi_minus"])
        oscillator = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        # Compute previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        prev_vortex = np.roll(oscillator, 1)
        prev_vortex[0] = np.nan
        # Define entry conditions
        # Long entry: k < oversold AND atr rising AND vortex negative
        oversold = params.get("stoch_rsi_oversold", 20)
        long_entry = (k < oversold) & (atr > np.roll(atr, 1)) & (oscillator < 0)
        # Short entry: k > overbought AND atr rising AND vortex positive
        overbought = params.get("stoch_rsi_overbought", 80)
        short_entry = (k > overbought) & (atr > np.roll(atr, 1)) & (oscillator > 0)
        # Define exit conditions
        # Exit long: k crosses above overbought OR vortex crosses above 0
        exit_long = (k > overbought) | (oscillator > 0)
        # Exit short: k crosses below overbought OR vortex crosses below 0
        exit_short = (k < overbought) | (oscillator < 0)
        # Set masks for entries and exits
        long_mask = long_entry
        short_mask = short_entry
        # Apply exits
        exit_long_mask = exit_long
        exit_short_mask = exit_short
        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Set exit signals
        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        close = df["close"].values
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)
        # Compute SL/TP for long entries
        entry_long = signals == 1.0
        if np.any(entry_long):
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        # Compute SL/TP for short entries
        entry_short = signals == -1.0
        if np.any(entry_short):
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
