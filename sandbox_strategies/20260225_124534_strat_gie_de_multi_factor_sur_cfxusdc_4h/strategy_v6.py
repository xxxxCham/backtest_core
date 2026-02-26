from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='supertrend_stoch_atr_multi_factor_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'stochastic', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_vol_threshold': 0.005,
         'leverage': 1,
         'stochastic_d_period': 3,
         'stochastic_k_period': 14,
         'stochastic_smooth_k': 3,
         'stop_atr_mult': 2.2,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 5.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stochastic_d_period': ParameterSpec(
                name='stochastic_d_period',
                min_val=2,
                max_val=10,
                default=3,
                param_type='int',
                step=1,
            ),
            'stochastic_smooth_k': ParameterSpec(
                name='stochastic_smooth_k',
                min_val=1,
                max_val=5,
                default=3,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.5,
                max_val=10.0,
                default=5.5,
                param_type='float',
                step=0.1,
            ),
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.001,
                max_val=0.02,
                default=0.005,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
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
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        close = df["close"].values

        # Supertrend direction
        st = indicators['supertrend']
        direction = np.nan_to_num(st["direction"])

        # Stochastic oscillator
        stoch = indicators['stochastic']
        k = np.nan_to_num(stoch["stoch_k"])
        d = np.nan_to_num(stoch["stoch_d"])

        # ATR
        atr = np.nan_to_num(indicators['atr'])

        # Stochastic cross detection
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan
        prev_d[0] = np.nan
        cross_up = (k > d) & (prev_k <= prev_d)
        cross_down = (k < d) & (prev_k >= prev_d)

        # Volatility filter
        vol_thresh = params.get("atr_vol_threshold", 0.005)
        vol_cond = atr > vol_thresh * close

        # Long entry: trend up, stochastic oversold & crossing up, sufficient volatility
        long_mask = (direction == 1) & (k < 20) & cross_up & vol_cond

        # Short entry: trend down, stochastic overbought & crossing down, sufficient volatility
        short_mask = (direction == -1) & (k > 80) & cross_down & vol_cond

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR‑based stop‑loss / take‑profit levels
        stop_mult = params.get("stop_atr_mult", 2.2)
        tp_mult = params.get("tp_atr_mult", 5.5)

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
