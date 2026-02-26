from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='volume_ema_momentum_0gusdc')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'volume_oscillator', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_fast': 50,
         'ema_slow': 200,
         'leverage': 1,
         'obv_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 2.0,
         'volume_oscillator_fast': 12,
         'volume_oscillator_slow': 26,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_fast': ParameterSpec(
                name='ema_fast',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'ema_slow': ParameterSpec(
                name='ema_slow',
                min_val=100,
                max_val=300,
                default=200,
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
                default=2.0,
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
        obv = np.nan_to_num(indicators['obv'])
        volume_oscillator = np.nan_to_num(indicators['volume_oscillator'])
        atr = np.nan_to_num(indicators['atr'])
        # Compute EMA arrays using params
        fast_ema = ema_fast
        slow_ema = ema_slow
        # Prepare rolling arrays
        obv_prev1 = np.roll(obv, 1)
        obv_prev1[0] = np.nan
        obv_prev2 = np.roll(obv, 2)
        obv_prev2[0] = np.nan
        obv_prev2[1] = np.nan
        vol_osc_prev1 = np.roll(volume_oscillator, 1)
        vol_osc_prev1[0] = np.nan
        vol_osc_prev2 = np.roll(volume_oscillator, 2)
        vol_osc_prev2[0] = np.nan
        vol_osc_prev2[1] = np.nan
        # Entry conditions
        # Long entry: EMA fast > EMA slow AND OBV rising AND Volume Oscillator increasing
        long_entry = (fast_ema > slow_ema) & (obv > obv_prev1) & (volume_oscillator > vol_osc_prev1) & (vol_osc_prev1 > vol_osc_prev2)
        # Short entry: EMA fast < EMA slow AND OBV falling AND Volume Oscillator decreasing
        short_entry = (fast_ema < slow_ema) & (obv < obv_prev1) & (volume_oscillator < vol_osc_prev1) & (vol_osc_prev1 < vol_osc_prev2)
        # Exit conditions
        # Exit long: EMA fast < EMA slow OR Volume Oscillator decreasing
        long_exit = (fast_ema < slow_ema) | (volume_oscillator < vol_osc_prev1)
        # Exit short: EMA fast > EMA slow OR Volume Oscillator increasing
        short_exit = (fast_ema > slow_ema) | (volume_oscillator > vol_osc_prev1)
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # Set long and short masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True
        # Apply exit conditions
        # For long positions, set exit where long_exit is True
        long_exit_mask = np.zeros(n, dtype=bool)
        long_exit_mask[long_exit] = True
        # For short positions, set exit where short_exit is True
        short_exit_mask = np.zeros(n, dtype=bool)
        short_exit_mask[short_exit] = True
        # Combine long and short entries
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Apply exit signals
        # If we're long and exit condition is met, set signal to flat
        signals[long_exit_mask & (signals == 1.0)] = 0.0
        # If we're short and exit condition is met, set signal to flat
        signals[short_exit_mask & (signals == -1.0)] = 0.0
        # Write ATR-based SL/TP levels into DataFrame
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan
        # On long entries, compute SL and TP
        close = df["close"].values
        entry_long = (signals == 1.0)
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]
        # On short entries, compute SL and TP
        entry_short = (signals == -1.0)
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
