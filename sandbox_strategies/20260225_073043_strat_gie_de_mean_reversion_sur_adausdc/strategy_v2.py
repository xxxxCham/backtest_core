from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mfi_stochrsi_mean_reversion_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['mfi', 'stoch_rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_min': 0.0005,
         'atr_period': 14,
         'leverage': 1,
         'mfi_neutral_high': 60,
         'mfi_neutral_low': 40,
         'mfi_overbought': 80,
         'mfi_oversold': 20,
         'mfi_period': 14,
         'stoch_rsi_k_period': 14,
         'stoch_rsi_overbought': 0.8,
         'stoch_rsi_oversold': 0.2,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.8,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'mfi_period': ParameterSpec(
                name='mfi_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stoch_rsi_k_period': ParameterSpec(
                name='stoch_rsi_k_period',
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.8,
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
        # Prepare arrays and parameters
        close = df["close"].values
        mfi = np.nan_to_num(indicators['mfi'])
        srsi = indicators['stoch_rsi']
        indicators['stoch_rsi']['k'] = np.nan_to_num(srsi["k"])
        atr = np.nan_to_num(indicators['atr'])

        mfi_oversold = params.get("mfi_oversold", 20)
        mfi_overbought = params.get("mfi_overbought", 80)
        stoch_rsi_oversold = params.get("stoch_rsi_oversold", 0.2)
        stoch_rsi_overbought = params.get("stoch_rsi_overbought", 0.8)
        atr_min = params.get("atr_min", 0.0005)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 2.8)

        # Previous values for cross detection
        prev_mfi = np.roll(mfi, 1)
        prev_mfi[0] = np.nan
        prev_srsi_k = np.roll(indicators['stoch_rsi']['k'], 1)
        prev_srsi_k[0] = np.nan

        # Entry conditions
        long_entry = (
            (mfi < mfi_oversold) &
            (prev_mfi >= mfi_oversold) &
            (indicators['stoch_rsi']['k'] < stoch_rsi_oversold) &
            (prev_srsi_k >= stoch_rsi_oversold) &
            (atr > atr_min)
        )

        short_entry = (
            (mfi > mfi_overbought) &
            (prev_mfi <= mfi_overbought) &
            (indicators['stoch_rsi']['k'] > stoch_rsi_overbought) &
            (prev_srsi_k <= stoch_rsi_overbought) &
            (atr > atr_min)
        )

        # Populate masks
        long_mask[:] = long_entry
        short_mask[:] = short_entry

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute SL/TP levels on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # Warmup protection (already zeroed but enforce)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
