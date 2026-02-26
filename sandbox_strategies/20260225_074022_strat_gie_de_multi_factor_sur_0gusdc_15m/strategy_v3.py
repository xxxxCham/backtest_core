from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='rsi_macd_adx_atr_multi_factor')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'macd', 'adx', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14,
         'leverage': 1,
         'macd_fast': 12,
         'macd_signal': 9,
         'macd_slow': 26,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.6,
         'tp_atr_mult': 3.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'macd_fast': ParameterSpec(
                name='macd_fast',
                min_val=5,
                max_val=20,
                default=12,
                param_type='int',
                step=1,
            ),
            'macd_slow': ParameterSpec(
                name='macd_slow',
                min_val=15,
                max_val=40,
                default=26,
                param_type='int',
                step=1,
            ),
            'macd_signal': ParameterSpec(
                name='macd_signal',
                min_val=5,
                max_val=15,
                default=9,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.6,
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
        # Prepare indicator arrays with NaN handling
        rsi = np.nan_to_num(indicators['rsi'])
        macd_dict = indicators['macd']
        indicators['macd']['macd'] = np.nan_to_num(macd_dict["macd"])
        indicators['macd']['signal'] = np.nan_to_num(macd_dict["signal"])
        adx_dict = indicators['adx']
        adx_val = np.nan_to_num(adx_dict["adx"])
        atr = np.nan_to_num(indicators['atr'])

        # Price series
        close = df["close"].values

        # Parameters with defaults
        stop_atr_mult = float(params.get("stop_atr_mult", 1.6))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.6))
        adx_entry_thr = float(params.get("adx_entry", 25))
        adx_exit_thr = float(params.get("adx_exit", 20))
        rsi_mid = float(params.get("rsi_mid", 50))

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        long_entry = (rsi > rsi_mid) & (indicators['macd']['macd'] > indicators['macd']['signal']) & (adx_val > adx_entry_thr)
        short_entry = (rsi < rsi_mid) & (indicators['macd']['macd'] < indicators['macd']['signal']) & (adx_val > adx_entry_thr)

        # Apply entries
        long_mask[long_entry] = True
        short_mask[short_entry] = True
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Cross detection for MACD
        prev_macd_line = np.roll(indicators['macd']['macd'], 1)
        prev_macd_signal = np.roll(indicators['macd']['signal'], 1)
        prev_macd_line[0] = np.nan
        prev_macd_signal[0] = np.nan

        cross_down = (indicators['macd']['macd'] < indicators['macd']['signal']) & (prev_macd_line >= prev_macd_signal)
        cross_up = (indicators['macd']['macd'] > indicators['macd']['signal']) & (prev_macd_line <= prev_macd_signal)

        # Exit conditions
        exit_long = ((cross_down) & (rsi < rsi_mid)) | (adx_val < adx_exit_thr)
        exit_short = ((cross_up) & (rsi > rsi_mid)) | (adx_val < adx_exit_thr)

        # Apply exits (override any entry on the same bar)
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # ATR‑based stop‑loss / take‑profit levels (only on entry bars)
        if np.any(long_mask):
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        if np.any(short_mask):
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
