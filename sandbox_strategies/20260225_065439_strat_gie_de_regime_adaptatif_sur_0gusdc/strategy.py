from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_vwap_atr_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'keltner_atr_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 3.2,
         'vol_mult': 1.0,
         'vwap_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_atr_mult': ParameterSpec(
                name='keltner_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'vwap_period': ParameterSpec(
                name='vwap_period',
                min_val=10,
                max_val=100,
                default=20,
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
            'vol_mult': ParameterSpec(
                name='vol_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=3.2,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=5,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays with NaN handling
        kelt = indicators['keltner']
        upper = np.nan_to_num(kelt["upper"])
        lower = np.nan_to_num(kelt["lower"])
        vwap_arr = np.nan_to_num(indicators['vwap'])
        atr_arr = np.nan_to_num(indicators['atr'])

        close = df["close"].values

        # Parameters
        vol_mult = float(params.get("vol_mult", 1.0))
        stop_atr_mult = float(params.get("stop_atr_mult", 1.1))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.2))

        # Volatility regime: high volatility if channel width exceeds ATR * vol_mult
        width = upper - lower
        high_vol = width > (atr_arr * vol_mult)
        low_vol = ~high_vol

        # Long entry conditions
        hv_long = high_vol & (close > upper)
        lv_long = low_vol & (close < vwap_arr) & ((vwap_arr - close) > (0.5 * atr_arr))
        long_mask = hv_long | lv_long

        # Short entry conditions
        hv_short = high_vol & (close < lower)
        lv_short = low_vol & (close > vwap_arr) & ((close - vwap_arr) > (0.5 * atr_arr))
        short_mask = hv_short | lv_short

        # Detect regime change (high_vol flag flips)
        prev_high_vol = np.roll(high_vol.astype(int), 1)
        prev_high_vol[0] = high_vol[0].astype(int)
        regime_change = high_vol != prev_high_vol

        # Suppress entries on regime change bars
        long_mask &= ~regime_change
        short_mask &= ~regime_change

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
            entry_price_long = close[long_mask]
            atr_long = atr_arr[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_atr_mult * atr_long
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_atr_mult * atr_long

        # Short SL/TP
        if short_mask.any():
            entry_price_short = close[short_mask]
            atr_short = atr_arr[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_atr_mult * atr_short
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_atr_mult * atr_short

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
