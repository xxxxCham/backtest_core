from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trx_usdc_30m_supertrend_macd_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'macd', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_vol_threshold': 0.5,
         'leverage': 1,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 5.9,
         'trailing_atr_mult': 2.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.1,
                max_val=2.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=5.9,
                param_type='float',
                step=0.1,
            ),
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
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

        # Extract indicators safely
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        macd_hist = np.nan_to_num(indicators['macd']["histogram"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        atr_vol_threshold = params.get("atr_vol_threshold", 0.5)
        stop_atr_mult = params.get("stop_atr_mult", 2.2)
        tp_atr_mult = params.get("tp_atr_mult", 5.9)

        # Entry conditions
        long_mask = (st_dir > 0) & (macd_hist > 0) & (atr > atr_vol_threshold)
        short_mask = (st_dir < 0) & (macd_hist < 0) & (atr > atr_vol_threshold)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions (not explicitly used as signals, but could be handled by stop/TP)
        # Prepare for potential exit logic if needed
        # cross detection for histogram crossing zero
        prev_hist = np.roll(macd_hist, 1)
        prev_hist[0] = np.nan
        cross_zero = (macd_hist > 0) & (prev_hist <= 0) | (macd_hist < 0) & (prev_hist >= 0)

        exit_long_mask = (st_dir < 0) | cross_zero | (atr < atr_vol_threshold)
        exit_short_mask = (st_dir > 0) | cross_zero | (atr < atr_vol_threshold)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR-based levels on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
