from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_vortex_ema_obv')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'vortex', 'atr', 'rsi', 'momentum']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 50,
         'leverage': 1,
         'obv_period': 20,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'obv_period': ParameterSpec(
                name='obv_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'vortex_period': ParameterSpec(
                name='vortex_period',
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

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        vortex = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        # Previous values for crossovers
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan
        prev_vortex = np.roll(vortex, 1)
        prev_vortex[0] = np.nan

        # Long entry: close > ema AND obv > prev_obv AND vortex > 0.5 AND vortex > prev_vortex
        long_entry = (close > ema) & (obv > prev_obv) & (vortex > 0.5) & (vortex > prev_vortex)
        long_mask = long_entry

        # Short entry: close < ema AND obv < prev_obv AND vortex > 0.5 AND vortex > prev_vortex
        short_entry = (close < ema) & (obv < prev_obv) & (vortex > 0.5) & (vortex > prev_vortex)
        short_mask = short_entry

        # Exit conditions
        # RSI > 70 or momentum < 0 for 3 consecutive periods
        rsi = np.nan_to_num(indicators['rsi'])
        momentum = np.nan_to_num(indicators['momentum'])
        prev_momentum = np.roll(momentum, 1)
        prev_momentum[0] = np.nan
        prev2_momentum = np.roll(momentum, 2)
        prev2_momentum[0] = np.nan
        prev2_momentum[1] = np.nan

        rsi_exit = rsi > 70
        momentum_exit = (momentum < 0) & (prev_momentum < 0) & (prev2_momentum < 0)
        exit_condition = rsi_exit | momentum_exit

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set SL/TP levels
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals