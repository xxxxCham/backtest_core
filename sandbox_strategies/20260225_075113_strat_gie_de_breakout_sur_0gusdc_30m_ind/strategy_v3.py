from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_pivot_atr_adx_30m_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['pivot_points', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_exit': 20,
         'adx_min': 25,
         'adx_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 6.0,
         'trailing_atr_mult': 2.0,
         'vol_thresh': 0.0015,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'vol_thresh': ParameterSpec(
                name='vol_thresh',
                min_val=0.0005,
                max_val=0.005,
                default=0.0015,
                param_type='float',
                step=0.1,
            ),
            'adx_min': ParameterSpec(
                name='adx_min',
                min_val=15,
                max_val=40,
                default=25,
                param_type='int',
                step=1,
            ),
            'adx_exit': ParameterSpec(
                name='adx_exit',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
                default=6.0,
                param_type='float',
                step=0.1,
            ),
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
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
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # Extract price series and indicators
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])

        pp = indicators['pivot_points']
        r1 = np.nan_to_num(pp["r1"])
        s1 = np.nan_to_num(pp["s1"])

        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # Strategy parameters
        vol_thresh = float(params.get("vol_thresh", 0.0015))
        adx_min = float(params.get("adx_min", 25))
        adx_exit = float(params.get("adx_exit", 20))
        stop_mult = float(params.get("stop_atr_mult", 1.0))
        tp_mult = float(params.get("tp_atr_mult", 6.0))

        # Entry conditions
        long_mask = (close > r1) & (atr > vol_thresh) & (adx_val > adx_min)
        short_mask = (close < s1) & (atr > vol_thresh) & (adx_val > adx_min)

        # Prevent consecutive same‑side entries
        prev_signal = np.roll(signals.values, 1)
        prev_signal[0] = 0.0
        long_mask = long_mask & (prev_signal != 1.0)
        short_mask = short_mask & (prev_signal != -1.0)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute ATR‑based stop‑loss and take‑profit levels
        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
