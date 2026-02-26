from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='atr_supertrend_vwap_adaptive_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'supertrend', 'vwap']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.0015,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 6.1,
         'trail_atr_mult': 2.4,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_threshold': ParameterSpec(
                name='atr_threshold',
                min_val=0.0005,
                max_val=0.003,
                default=0.0015,
                param_type='float',
                step=0.0001,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=4.0,
                max_val=8.0,
                default=6.1,
                param_type='float',
                step=0.1,
            ),
            'trail_atr_mult': ParameterSpec(
                name='trail_atr_mult',
                min_val=1.5,
                max_val=4.0,
                default=2.4,
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

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        atr = np.nan_to_num(indicators['atr'])
        st = np.nan_to_num(indicators['supertrend']["supertrend"])
        vwap = np.nan_to_num(indicators['vwap'])
        close = df["close"].values

        # Volatility regimes
        high_vol = atr > params["atr_threshold"]
        low_vol = ~high_vol

        # Long conditions
        long_high = high_vol & (close > st)
        long_low = low_vol & (close < vwap) & (close > st)
        long_mask = long_high | long_low

        # Short conditions
        short_high = high_vol & (close < st)
        short_low = low_vol & (close > vwap) & (close < st)
        short_mask = short_high | short_low

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR‑based SL/TP for long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        # ATR‑based SL/TP for short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
