from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='atr_supertrend_vwap_adaptive_30m_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'supertrend', 'vwap']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_threshold': 0.0015,
         'leverage': 1,
         'stop_atr_mult': 2.4,
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
                max_val=4.0,
                default=2.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=10.0,
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
        # Extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        vwap = np.nan_to_num(indicators['vwap'])
        st_line = np.nan_to_num(indicators['supertrend']["supertrend"])

        # Parameters
        atr_threshold = params.get("atr_threshold", 0.0015)
        stop_atr_mult = params.get("stop_atr_mult", 2.4)
        tp_atr_mult = params.get("tp_atr_mult", 6.1)

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Breakout entry when ATR high
        breakout_long = (atr > atr_threshold) & (close > st_line)
        breakout_short = (atr > atr_threshold) & (close < st_line)

        # Mean‑reversion entry when ATR low
        mean_rev_long = (atr <= atr_threshold) & (close > vwap)
        mean_rev_short = (atr <= atr_threshold) & (close < vwap)

        long_mask = breakout_long | mean_rev_long
        short_mask = breakout_short | mean_rev_short

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR based SL/TP levels on entry bars
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        long_entry = signals == 1.0
        short_entry = signals == -1.0

        df.loc[long_entry, "bb_stop_long"] = close[long_entry] - stop_atr_mult * atr[long_entry]
        df.loc[long_entry, "bb_tp_long"] = close[long_entry] + tp_atr_mult * atr[long_entry]

        df.loc[short_entry, "bb_stop_short"] = close[short_entry] + stop_atr_mult * atr[short_entry]
        df.loc[short_entry, "bb_tp_short"] = close[short_entry] - tp_atr_mult * atr[short_entry]
        signals.iloc[:warmup] = 0.0
        return signals
