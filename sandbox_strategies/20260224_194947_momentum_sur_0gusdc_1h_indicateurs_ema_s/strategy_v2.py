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
        return ['ema', 'stochastic', 'volume_oscillator', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stochastic_d': 3,
         'stochastic_k': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 1.5,
         'volume_oscillator_period': 14,
         'warmup': 100}

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
            'stochastic_k': ParameterSpec(
                name='stochastic_k',
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
        ema_fast = np.nan_to_num(indicators['ema'])
        ema_slow = np.nan_to_num(indicators['ema'])
        stochastic = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(indicators['stochastic']["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(indicators['stochastic']["stoch_d"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        keltner = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(indicators['keltner']["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(indicators['keltner']["lower"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values
        # Compute EMA arrays for crossover detection
        ema_fast_array = ema_fast
        ema_slow_array = ema_slow
        # Compute previous arrays for crossovers
        prev_ema_fast = np.roll(ema_fast_array, 1)
        prev_ema_slow = np.roll(ema_slow_array, 1)
        prev_ema_fast[0] = np.nan
        prev_ema_slow[0] = np.nan
        # Long crossover condition
        ema_cross_up = (ema_fast_array > ema_slow_array) & (prev_ema_fast <= prev_ema_slow)
        # Short crossover condition
        ema_cross_down = (ema_fast_array < ema_slow_array) & (prev_ema_fast >= prev_ema_slow)
        # Stochastic conditions
        stoch_long_cond = indicators['stochastic']['stoch_k'] > 80
        stoch_short_cond = indicators['stochastic']['stoch_k'] < 20
        # Volume oscillator conditions
        vol_long_cond = volume_osc > 0
        vol_short_cond = volume_osc < 0
        # Keltner regime filter
        close_above_kelt_upper = close > indicators['keltner']['upper']
        close_below_kelt_lower = close < indicators['keltner']['lower']
        # Long entry: EMA crossover up + stochastic > 80 + volume > 0 + in uptrend
        long_entry_cond = ema_cross_up & stoch_long_cond & vol_long_cond & close_above_kelt_upper
        long_mask = long_entry_cond
        # Short entry: EMA crossover down + stochastic < 20 + volume < 0 + in downtrend
        short_entry_cond = ema_cross_down & stoch_short_cond & vol_short_cond & close_below_kelt_lower
        short_mask = short_entry_cond
        # Exit conditions
        exit_long = ema_cross_down | (indicators['stochastic']['stoch_k'] < 20)
        exit_short = ema_cross_up | (indicators['stochastic']['stoch_k'] > 80)
        # Apply long signals
        signals[long_mask] = 1.0
        # Apply short signals
        signals[short_mask] = -1.0
        # Risk management
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 1.5)
        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # Long SL/TP
        entry_long = signals == 1.0
        if entry_long.any():
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        # Short SL/TP
        entry_short = signals == -1.0
        if entry_short.any():
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
