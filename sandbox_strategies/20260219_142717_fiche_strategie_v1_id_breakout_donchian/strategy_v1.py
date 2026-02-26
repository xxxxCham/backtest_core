from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='breakout_donchian_adx_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['donchian', 'adx', 'atr', 'sma']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 4.0,
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
                max_val=10.0,
                default=4.0,
                param_type='float',
                step=0.1,
            ),
            'adx_period': ParameterSpec(
                name='adx_period',
                min_val=10,
                max_val=50,
                default=23,
                param_type='int',
                step=1,
            ),
            'donchian_period': ParameterSpec(
                name='donchian_period',
                min_val=10,
                max_val=100,
                default=40,
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
        # Extract indicator arrays with np.nan_to_num
        close = df["close"].values
        don = indicators['donchian']
        don_upper = np.nan_to_num(don["upper"])
        don_lower = np.nan_to_num(don["lower"])
        don_middle = np.nan_to_num(don["middle"])
        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])
        sma = np.nan_to_num(indicators['sma'])
        atr = np.nan_to_num(indicators['atr'])

        # Define entry masks
        long_mask = (close > don_upper) & (adx_val > 30) & (close > sma)
        short_mask = (close < don_lower) & (adx_val > 30) & (close < sma)

        # Helper for cross_any
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(don_middle, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        cross_up = (close > don_middle) & (prev_close <= prev_middle)
        cross_down = (close < don_middle) & (prev_close >= prev_middle)
        cross_any = cross_up | cross_down

        # Exit mask
        exit_mask = cross_any | (adx_val < 20)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based risk management levels on entry bars
        stop_mult = float(params.get("stop_atr_mult", 1.5))
        tp_mult = float(params.get("tp_atr_mult", 4.0))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
