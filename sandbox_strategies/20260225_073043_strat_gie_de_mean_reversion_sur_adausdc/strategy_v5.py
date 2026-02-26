from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mfi_stochrsi_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['mfi', 'stoch_rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'mfi_overbought': 80,
         'mfi_oversold': 20,
         'stoch_rsi_overbought': 80,
         'stoch_rsi_oversold': 20,
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
            'stoch_rsi_period': ParameterSpec(
                name='stoch_rsi_period',
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
                max_val=3.0,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # warmup protection
        signals.iloc[:50] = 0.0

        # extract indicators with NaN handling
        mfi = np.nan_to_num(indicators['mfi'])
        srsi = indicators['stoch_rsi']
        indicators['stoch_rsi']['k'] = np.nan_to_num(srsi["k"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # parameters (fallback defaults)
        mfi_oversold = params.get("mfi_oversold", 20)
        mfi_overbought = params.get("mfi_overbought", 80)
        srsi_oversold = params.get("stoch_rsi_oversold", 20)
        srsi_overbought = params.get("stoch_rsi_overbought", 80)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 2.8)
        mfi_exit_low = params.get("mfi_exit_low", 45)
        mfi_exit_high = params.get("mfi_exit_high", 55)

        # entry conditions
        long_mask = (mfi < mfi_oversold) & (indicators['stoch_rsi']['k'] < srsi_oversold)
        short_mask = (mfi > mfi_overbought) & (indicators['stoch_rsi']['k'] > srsi_overbought)

        # assign entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # exit condition: neutral MFI zone
        exit_mask = (mfi >= mfi_exit_low) & (mfi <= mfi_exit_high)
        signals[exit_mask] = 0.0

        # initialize SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # write SL/TP levels only on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
