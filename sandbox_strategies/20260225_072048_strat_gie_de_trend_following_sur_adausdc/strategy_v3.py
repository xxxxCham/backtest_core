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
         'atr_vol_threshold': 0.0005,
         'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 1.0,
         'tp_atr_mult': 2.7,
         'vortex_period': 14,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=20,
                max_val=200,
                default=50,
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
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_vol_threshold': ParameterSpec(
                name='atr_vol_threshold',
                min_val=0.0001,
                max_val=0.01,
                default=0.0005,
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
                max_val=3,
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
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract price and indicators, safely handling NaNs
        close = np.nan_to_num(df["close"].values)

        sma = np.nan_to_num(indicators['sma'])

        vx = indicators['vortex']
        indicators['vortex']['vi_plus'] = np.nan_to_num(vx["vi_plus"])
        indicators['vortex']['vi_minus'] = np.nan_to_num(vx["vi_minus"])

        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        atr_vol_threshold = float(params.get("atr_vol_threshold", 0.0005))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.0))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.7))

        # Entry conditions
        long_entry = (indicators['vortex']['vi_plus'] > indicators['vortex']['vi_minus']) & (close > sma) & (atr > atr_vol_threshold)
        short_entry = (indicators['vortex']['vi_minus'] > indicators['vortex']['vi_plus']) & (close < sma) & (atr > atr_vol_threshold)

        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Assign entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions (reverse of entry)
        long_exit = (indicators['vortex']['vi_plus'] <= indicators['vortex']['vi_minus']) | (close < sma)
        short_exit = (indicators['vortex']['vi_minus'] <= indicators['vortex']['vi_plus']) | (close > sma)

        # Override signals to flat on exit bars
        signals[long_exit] = 0.0
        signals[short_exit] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare ATR‑based stop‑loss / take‑profit columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long SL/TP
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Short SL/TP
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
