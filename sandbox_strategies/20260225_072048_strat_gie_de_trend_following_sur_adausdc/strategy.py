from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vortex_sma_atr_trend_adausdc_1h')

    @property
    def required_indicators(self) -> List[str]:
        return ['vortex', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_vol_threshold': 0.001,
         'leverage': 1,
         'sma_period': 100,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.7,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'vortex_period': ParameterSpec(
                name='vortex_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=50,
                max_val=200,
                default=100,
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
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0005,
                max_val=0.01,
                default=0.001,
                param_type='float',
                step=0.1,
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
                min_val=1.0,
                max_val=5.0,
                default=2.7,
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
        # Initialise masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract price series and indicators, ensuring NaNs are replaced
        close = df["close"].values
        sma = np.nan_to_num(indicators['sma'])
        vx = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vx["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vx["vi_minus"])
        atr = np.nan_to_num(indicators['atr'])

        # Parameter thresholds
        atr_vol_thr = float(params.get("atr_vol_threshold", 0.001))
        stop_mult = float(params.get("stop_atr_mult", 1.0))
        tp_mult = float(params.get("tp_atr_mult", 2.7))

        # Entry conditions
        long_entry = (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (indicators['vortex']['vi_plus'] > 1.0) & (close > sma) & (atr > atr_vol_thr)
        short_entry = (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus']) & (indicators['vortex']['vi_minus'] > 1.0) & (close < sma) & (atr > atr_vol_thr)

        long_mask = long_entry
        short_mask = short_entry

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Prepare ATR‑based stop‑loss and take‑profit columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Long SL/TP
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
