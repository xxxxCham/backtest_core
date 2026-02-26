from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='cci_stoch_rsi_mfi_mean_rev_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'stoch_rsi', 'mfi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'cci_period': 20,
            'leverage': 1,
            'mfi_overbought': 70,
            'mfi_oversold': 30,
            'stoch_rsi_overbought': 80,
            'stoch_rsi_oversold': 20,
            'stoch_rsi_period': 14,
            'stop_atr_mult': 1.2,
            'tp_atr_mult': 2.5,
            'warmup': 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_period': ParameterSpec(
                name='cci_period',
                min_val=10,
                max_val=30,
                default=20,
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
            'stoch_rsi_overbought': ParameterSpec(
                name='stoch_rsi_overbought',
                min_val=70,
                max_val=90,
                default=80,
                param_type='int',
                step=1,
            ),
            'stoch_rsi_oversold': ParameterSpec(
                name='stoch_rsi_oversold',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'mfi_overbought': ParameterSpec(
                name='mfi_overbought',
                min_val=60,
                max_val=90,
                default=70,
                param_type='int',
                step=1,
            ),
            'mfi_oversold': ParameterSpec(
                name='mfi_oversold',
                min_val=10,
                max_val=40,
                default=30,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.5,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=100,
                default=50,
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

    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        warmup = int(params.get('warmup', 50))

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Prepare indicator arrays
        cci = np.nan_to_num(indicators['cci'])
        atr = np.nan_to_num(indicators['atr'])
        mfi = np.nan_to_num(indicators['mfi'])
        # stoch_rsi contains dict with keys 'k', 'd', 'signal'
        stoch_k = np.nan_to_num(indicators['stoch_rsi']['k'])

        # Masks for entries
        long_mask = (cci < -100) & (stoch_k < params["stoch_rsi_oversold"]) & (mfi < params["mfi_oversold"])
        short_mask = (cci > 100) & (stoch_k > params["stoch_rsi_overbought"]) & (mfi > params["mfi_overbought"])

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions using cross detection
        prev_cci = np.roll(cci, 1)
        prev_cci[0] = np.nan
        exit_long_mask = (cci > -20) & (prev_cci <= -20)
        exit_short_mask = (cci < 20) & (prev_cci >= 20)

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Ensure warmup period is flat
        signals.iloc[:warmup] = 0.0

        # ATR-based stop/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = params.get("stop_atr_mult", 1.2)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        close = df["close"].values
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals