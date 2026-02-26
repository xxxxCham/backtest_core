from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_frontusdc_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'volume_oscillator', 'williams_r', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 20,
         'ema_slow': 50,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50,
         'williams_r_overbought': -20,
         'williams_r_oversold': -80}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=10,
                max_val=30,
                default=20,
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
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        close = df["close"].values
        ema_fast = np.nan_to_num(indicators['ema'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        williams_r = np.nan_to_num(indicators['williams_r'])
        atr = np.nan_to_num(indicators['atr'])

        # Compute EMA of volume oscillator
        ema_volume_oscillator = np.nan_to_num(pd.Series(volume_oscillator).rolling(20).mean().values)

        # Entry conditions
        # Long entry: close touches lower EMA band AND volume_oscillator > EMA of volume_oscillator AND williams_r < -80
        close_touches_lower = np.isclose(close, ema_fast, atol=1e-6)
        volume_above_ema = volume_oscillator > ema_volume_oscillator
        williams_r_below_oversold = williams_r < params["williams_r_oversold"]
        long_entry_condition = close_touches_lower & volume_above_ema & williams_r_below_oversold

        # Short entry: close touches upper EMA band AND volume_oscillator > EMA of volume_oscillator AND williams_r > -80
        close_touches_upper = np.isclose(close, ema_fast, atol=1e-6)
        williams_r_above_oversold = williams_r > params["williams_r_oversold"]
        short_entry_condition = close_touches_upper & volume_above_ema & williams_r_above_oversold

        # Exit conditions
        # Exit long: close crosses above EMA middle band OR williams_r > -20
        prev_ema_fast = np.roll(ema_fast, 1)
        prev_ema_fast[0] = np.nan
        cross_above_ema = (close > ema_fast) & (prev_ema_fast <= ema_fast)
        williams_r_above_overbought = williams_r > params["williams_r_overbought"]
        long_exit_condition = cross_above_ema | williams_r_above_overbought

        # Exit short: close crosses below EMA middle band OR williams_r < -20
        cross_below_ema = (close < ema_fast) & (prev_ema_fast >= ema_fast)
        short_exit_condition = cross_below_ema | williams_r_above_overbought

        # Apply entry signals
        long_mask = long_entry_condition
        short_mask = short_entry_condition

        # Apply exit signals
        long_exit_mask = long_exit_condition
        short_exit_mask = short_exit_condition

        # Avoid overlapping signals
        # Long signal overrides any short signal
        # Short signal overrides any long signal
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # For ATR-based risk management
        # Initialize SL/TP columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # On long entry bars, compute ATR-based levels
        entry_long_mask = (signals == 1.0)
        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - params["stop_atr_mult"] * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + params["tp_atr_mult"] * atr[entry_long_mask]

        # On short entry bars, compute ATR-based levels
        entry_short_mask = (signals == -1.0)
        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + params["stop_atr_mult"] * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - params["tp_atr_mult"] * atr[entry_short_mask]

        # Apply exit signals
        # Long exit
        signals[long_exit_mask] = 0.0
        # Short exit
        signals[short_exit_mask] = 0.0
        signals.iloc[:warmup] = 0.0
        return signals
