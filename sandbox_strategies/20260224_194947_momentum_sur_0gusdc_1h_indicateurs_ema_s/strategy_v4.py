from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ema_stochastic_volume_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'stochastic', 'volume_oscillator', 'atr', 'keltner']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'leverage': 1,
         'stoch_d': 3,
         'stoch_k': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 1.5,
         'volume_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=5,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=30,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0
        # extract indicators
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])
        vol_osc = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        close = df["close"].values
        # prepare EMA arrays
        ema_fast = ema_fast.reshape(-1)
        ema_slow = ema_slow.reshape(-1)
        # compute crossovers
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        cross_up = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        cross_down = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)
        # entry conditions
        long_entry = cross_up & (indicators['stochastic']['stoch_k'] > 80) & (vol_osc > 0)
        short_entry = cross_down & (indicators['stochastic']['stoch_k'] < 20) & (vol_osc < 0)
        # exit conditions
        long_exit = cross_down | (indicators['stochastic']['stoch_k'] < 20)
        short_exit = cross_up | (indicators['stochastic']['stoch_k'] > 80)
        # volatility regime filter
        regime_filter_long = close > indicators['keltner']['upper']
        regime_filter_short = close < indicators['keltner']['lower']
        long_entry = long_entry & regime_filter_long
        short_entry = short_entry & regime_filter_short
        # apply masks
        long_mask = long_entry
        short_mask = short_entry
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # exit signals
        exit_long = long_exit & (np.roll(signals, 1) == 1.0)
        exit_short = short_exit & (np.roll(signals, 1) == -1.0)
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0
        # ATR-based risk management
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # long stop/tp
        entry_long_mask = signals == 1.0
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]
        # short stop/tp
        entry_short_mask = signals == -1.0
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
