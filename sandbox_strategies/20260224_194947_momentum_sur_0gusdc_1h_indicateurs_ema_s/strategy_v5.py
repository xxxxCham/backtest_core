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
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stoch_overbought': 80,
         'stoch_oversold': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'volume_oscillator_period': 12,
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
            'stoch_overbought': ParameterSpec(
                name='stoch_overbought',
                min_val=70,
                max_val=95,
                default=80,
                param_type='int',
                step=1,
            ),
            'stoch_oversold': ParameterSpec(
                name='stoch_oversold',
                min_val=5,
                max_val=30,
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
        indicators['keltner']['middle'] = np.nan_to_num(indicators['keltner']["middle"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])

        # EMA crossover signals
        ema_fast_short = ema_fast[:len(ema_fast) - params["ema_fast"] + 1]
        ema_slow_short = ema_slow[:len(ema_slow) - params["ema_slow"] + 1]
        ema_fast_long = ema_fast[params["ema_fast"] - 1:]
        ema_slow_long = ema_slow[params["ema_slow"] - 1:]

        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_slow = np.roll(ema_slow, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan

        cross_up_ema = (ema_fast > ema_slow) & (prev_ema_fast <= prev_ema_slow)
        cross_down_ema = (ema_fast < ema_slow) & (prev_ema_fast >= prev_ema_slow)

        # Entry conditions
        long_entry = cross_up_ema & (indicators['stochastic']['stoch_k'] > params["stoch_overbought"]) & (volume_osc > 0)
        short_entry = cross_down_ema & (indicators['stochastic']['stoch_k'] < params["stoch_oversold"]) & (volume_osc < 0)

        # Exit conditions
        exit_long = cross_down_ema | (indicators['stochastic']['stoch_k'] < params["stoch_oversold"])
        exit_short = cross_up_ema | (indicators['stochastic']['stoch_k'] > params["stoch_overbought"])

        # Volatility regime filter via Keltner Channel
        vol_regime = (indicators['keltner']['upper'] - indicators['keltner']['lower']) / indicators['keltner']['middle'] > 0.01

        # Apply long/short masks
        long_mask = long_entry & vol_regime
        short_mask = short_entry & vol_regime

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set exit signals
        exit_long_mask = exit_long & (np.roll(signals, 1) == 1.0)
        exit_short_mask = exit_short & (np.roll(signals, 1) == -1.0)

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP levels
        close = df["close"].values
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
