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
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
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
        overbought = params.get("stoch_rsi_overbought", 80)
        oversold = params.get("stoch_rsi_oversold", 20)

        vortex = indicators['vortex']
        oscillator = np.nan_to_num(indicators['vortex']["oscillator"])

        atr = np.nan_to_num(indicators['atr'])

        # Previous values for crossovers
        prev_k = np.roll(k, 1)
        prev_k[0] = np.nan
        prev_d = np.roll(d, 1)
        prev_d[0] = np.nan
        prev_oscillator = np.roll(oscillator, 1)
        prev_oscillator[0] = np.nan
        prev_atr = np.roll(atr, 1)
        prev_atr[0] = np.nan

        # Entry conditions
        # Long entry: k crosses below oversold, vortex oscillator < 0, atr increasing
        long_entry = (k < oversold) & (prev_k >= oversold) & (oscillator < 0) & (atr > prev_atr)
        long_mask = long_mask | long_entry

        # Short entry: k crosses above overbought, vortex oscillator > 0, atr increasing
        short_entry = (k > overbought) & (prev_k <= overbought) & (oscillator > 0) & (atr > prev_atr)
        short_mask = short_mask | short_entry

        # Exit conditions
        # Exit long: k crosses above overbought or oscillator crosses above 0
        long_exit = (k > overbought) & (prev_k <= overbought) | (oscillator > 0) & (prev_oscillator <= 0)
        long_mask = long_mask & ~long_exit

        # Exit short: k crosses below overbought or oscillator crosses below 0
        short_exit = (k < overbought) & (prev_k >= overbought) | (oscillator < 0) & (prev_oscillator >= 0)
        short_mask = short_mask & ~short_exit

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        close = df["close"].values
        entry_long = signals == 1.0
        entry_short = signals == -1.0

        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
