from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_ema_obv_momentum')

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'obv', 'vortex', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'ema_period': 50,
         'leverage': 1,
         'obv_period': 20,
         'rsi_overbought': 70,
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
        # warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        close = df["close"].values
        ema = np.nan_to_num(indicators['ema'])
        obv = np.nan_to_num(indicators['obv'])
        vortex = np.nan_to_num(indicators['vortex']["oscillator"])
        atr = np.nan_to_num(indicators['atr'])
        # Remove rsi reference since it's not in required_indicators
        # rsi = np.nan_to_num(indicators['rsi'])

        # Previous values for crossovers
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan
        prev_vortex = np.roll(vortex, 1)
        prev_vortex[0] = np.nan

        # Entry conditions
        ema_long = close > ema
        obv_long = obv > prev_obv
        vortex_long = (vortex > 0.5) & (vortex > prev_vortex)
        long_condition = ema_long & obv_long & vortex_long

        ema_short = close < ema
        obv_short = obv < prev_obv
        vortex_short = (vortex > 0.5) & (vortex > prev_vortex)
        short_condition = ema_short & obv_short & vortex_short

        long_mask = long_condition
        short_mask = short_condition

        # Exit conditions
        # Remove rsi reference since it's not in required_indicators
        # rsi_overbought = rsi > params["rsi_overbought"]

        # Momentum filter
        momentum = np.diff(close)
        momentum = np.insert(momentum, 0, 0.0)
        momentum = np.nan_to_num(momentum)
        prev_momentum = np.roll(momentum, 1)
        prev_momentum[0] = 0.0
        prev2_momentum = np.roll(momentum, 2)
        prev2_momentum[0] = 0.0
        prev2_momentum[1] = 0.0

        momentum_down = (momentum < 0) & (prev_momentum < 0) & (prev2_momentum < 0)
        # Remove rsi reference since it's not in required_indicators
        # exit_condition = rsi_overbought | momentum_down
        exit_condition = momentum_down

        # Apply exits to existing positions
        long_signal_mask = long_mask
        short_signal_mask = short_mask

        # For simplicity, assume no position tracking, just apply entry signals
        signals[long_signal_mask] = 1.0
        signals[short_signal_mask] = -1.0

        # Set ATR-based stop-loss and take-profit
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # On long entries
        entry_mask_long = (signals == 1.0)
        df.loc[entry_mask_long, "bb_stop_long"] = close[entry_mask_long] - params["stop_atr_mult"] * atr[entry_mask_long]
        df.loc[entry_mask_long, "bb_tp_long"] = close[entry_mask_long] + params["tp_atr_mult"] * atr[entry_mask_long]

        # On short entries
        entry_mask_short = (signals == -1.0)
        df.loc[entry_mask_short, "bb_stop_short"] = close[entry_mask_short] + params["stop_atr_mult"] * atr[entry_mask_short]
        df.loc[entry_mask_short, "bb_tp_short"] = close[entry_mask_short] - params["tp_atr_mult"] * atr[entry_mask_short]
        signals.iloc[:warmup] = 0.0
        return signals