from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_0gusdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'volume_oscillator', 'atr', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stoch_d': 3,
         'stoch_k': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_long': 26,
         'volume_short': 12,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'stoch_k': ParameterSpec(
                name='stoch_k',
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

        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        stochastic = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(indicators['stochastic']["stoch_d"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        keltner = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(indicators['keltner']["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])
        close = df["close"].values

        # EMA crossovers
        ema_fast_vals = ema_fast
        ema_slow_vals = ema_slow

        prev_ema_fast = np.roll(ema_fast_vals, 1)
        prev_ema_slow = np.roll(ema_slow_vals, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan

        cross_up = (ema_fast_vals > ema_slow_vals) & (prev_ema_fast <= prev_ema_slow)
        cross_down = (ema_fast_vals < ema_slow_vals) & (prev_ema_fast >= prev_ema_slow)

        # Entry conditions
        long_entry = cross_up & (indicators['stochastic']['stoch_k'] > 80) & (volume_osc > 0)
        short_entry = cross_down & (indicators['stochastic']['stoch_k'] < 20) & (volume_osc < 0)

        # Volatility regime filter via Keltner Channel
        regime_filter_long = close > indicators['keltner']['upper']
        regime_filter_short = close < indicators['keltner']['lower']

        long_mask = long_entry & regime_filter_long
        short_mask = short_entry & regime_filter_short

        # Exit conditions
        exit_long = cross_down | (indicators['stochastic']['stoch_k'] < 20)
        exit_short = cross_up | (indicators['stochastic']['stoch_k'] > 80)

        # Apply exits to existing positions
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)

        # Combine all masks
        long_mask = long_mask | exit_long_mask
        short_mask = short_mask | exit_short_mask

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long_mask = (signals == 1.0)
        entry_short_mask = (signals == -1.0)

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
