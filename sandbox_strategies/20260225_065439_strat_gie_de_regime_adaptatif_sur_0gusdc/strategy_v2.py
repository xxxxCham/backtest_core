from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_keltner_vwap_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'vwap', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 3.2,
         'vol_factor': 1.0,
         'vwap_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_mult': ParameterSpec(
                name='keltner_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'vwap_period': ParameterSpec(
                name='vwap_period',
                min_val=5,
                max_val=100,
                default=20,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'vol_factor': ParameterSpec(
                name='vol_factor',
                min_val=0.5,
                max_val=3.0,
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=5.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=3.2,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=10,
                default=1,
                param_type='int',
                step=1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=200,
                default=50,
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
        # Extract price series
        close = df["close"].values

        # Indicator arrays with NaN handling
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        vwap_arr = np.nan_to_num(indicators['vwap'])
        atr_arr = np.nan_to_num(indicators['atr'])

        # Parameters
        vol_factor = params.get("vol_factor", 1.0)
        stop_mult = params.get("stop_atr_mult", 1.1)
        tp_mult = params.get("tp_atr_mult", 3.2)
        warmup = int(params.get("warmup", 50))

        # High‑volatility regime detection
        high_vol = (indicators['keltner']['upper'] - indicators['keltner']['lower']) > atr_arr * vol_factor

        # VWAP cross detection
        prev_close = np.roll(close, 1)
        prev_vwap = np.roll(vwap_arr, 1)
        prev_close[0] = np.nan
        prev_vwap[0] = np.nan
        cross_up_vwap = (close > vwap_arr) & (prev_close <= prev_vwap)
        cross_down_vwap = (close < vwap_arr) & (prev_close >= prev_vwap)

        # Entry masks
        long_mask = (high_vol & (close > indicators['keltner']['upper'])) | (~high_vol & cross_up_vwap)
        short_mask = (high_vol & (close < indicators['keltner']['lower'])) | (~high_vol & cross_down_vwap)

        # Regime‑change detection for exit (no new entry on regime‑change bar)
        prev_high_vol = np.roll(high_vol.astype(float), 1)
        prev_high_vol[0] = np.nan
        regime_change = (high_vol != prev_high_vol) & (~np.isnan(prev_high_vol))
        long_mask &= ~regime_change
        short_mask &= ~regime_change

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        if warmup > 0:
            signals.iloc[:warmup] = 0.0

        # Initialise SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR‑based stop‑loss and take‑profit levels on entry bars
        if long_mask.any():
            entry_price_long = close[long_mask]
            atr_long = atr_arr[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - stop_mult * atr_long
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + tp_mult * atr_long

        if short_mask.any():
            entry_price_short = close[short_mask]
            atr_short = atr_arr[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + stop_mult * atr_short
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - tp_mult * atr_short

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
