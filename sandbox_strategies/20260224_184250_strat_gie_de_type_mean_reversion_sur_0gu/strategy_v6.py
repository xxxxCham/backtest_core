from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_supertrend_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'volume_oscillator', 'supertrend', 'donchian', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'bollinger_bandwidth_threshold': 0.02,
         'bollinger_period': 20,
         'bollinger_std_dev': 2,
         'donchian_period': 20,
         'ema_period': 20,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stop_atr_mult': 1.5,
         'supertrend_multiplier': 3,
         'supertrend_period': 10,
         'tp_atr_mult': 3.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'bollinger_bandwidth_threshold': ParameterSpec(
                name='bollinger_bandwidth_threshold',
                min_val=0.005,
                max_val=0.1,
                default=0.02,
                param_type='float',
                step=0.1,
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
                default=3.0,
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
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        volume_osc = np.nan_to_num(indicators['volume_oscillator'])
        st = indicators['supertrend']
        st_direction = np.nan_to_num(st["direction"])
        dc = indicators['donchian']
        indicators['donchian']['upper'] = np.nan_to_num(dc["upper"])
        indicators['donchian']['lower'] = np.nan_to_num(dc["lower"])
        stoch = indicators['stochastic']
        indicators['stochastic']['stoch_k'] = np.nan_to_num(stoch["stoch_k"])
        indicators['stochastic']['stoch_d'] = np.nan_to_num(stoch["stoch_d"])
        atr = np.nan_to_num(indicators['atr'])
        # Compute bandwidth
        bb_bandwidth = (indicators['bollinger']['upper'] - indicators['bollinger']['lower']) / indicators['bollinger']['middle']
        # Compute crosses for EMA
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_ema = np.roll(ema, 1)
        prev_ema[0] = np.nan
        cross_up_ema = (close > ema) & (prev_close <= prev_ema)
        cross_down_ema = (close < ema) & (prev_close >= prev_ema)
        # Compute volume condition
        volume_mean = np.mean(volume_osc)
        vol_condition_long = volume_osc > volume_mean
        vol_condition_short = volume_osc < volume_mean
        # Compute trend filter using Supertrend and Donchian
        # Long entry if price is below supertrend direction and in range (Donchian lower)
        trend_long_filter = (st_direction == 1.0) & (close > indicators['donchian']['lower'])
        trend_short_filter = (st_direction == -1.0) & (close < indicators['donchian']['upper'])
        # Entry conditions
        long_condition = (cross_up_ema) & (bb_bandwidth < params["bollinger_bandwidth_threshold"]) & (vol_condition_long) & (trend_long_filter)
        short_condition = (cross_down_ema) & (bb_bandwidth < params["bollinger_bandwidth_threshold"]) & (vol_condition_short) & (trend_short_filter)
        # Stochastic confirmation
        prev_stoch_k = np.roll(indicators['stochastic']['stoch_k'], 1)
        prev_stoch_k[0] = np.nan
        stoch_cross_down = (indicators['stochastic']['stoch_k'] < 20) & (prev_stoch_k >= 20)
        stoch_cross_up = (indicators['stochastic']['stoch_k'] > 80) & (prev_stoch_k <= 80)
        # Apply confirmation
        long_mask = long_condition & stoch_cross_down
        short_mask = short_condition & stoch_cross_up
        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Warmup protection
        signals.iloc[:warmup] = 0.0
        # ATR-based SL/TP
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        entry_long = (signals == 1.0)
        entry_short = (signals == -1.0)
        if np.any(entry_long):
            df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
            df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        if np.any(entry_short):
            df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
            df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
