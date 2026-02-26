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
        return {'atr_min': 0.0005,
         'atr_period': 14,
         'leverage': 1,
         'mfi_neutral_high': 60,
         'mfi_neutral_low': 40,
         'mfi_overbought': 80,
         'mfi_oversold': 20,
         'mfi_period': 14,
         'stoch_rsi_overbought': 0.8,
         'stoch_rsi_oversold': 0.2,
         'stoch_rsi_period': 14,
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
            'mfi_oversold': ParameterSpec(
                name='mfi_oversold',
                min_val=10,
                max_val=30,
                default=20,
                param_type='int',
                step=1,
            ),
            'mfi_overbought': ParameterSpec(
                name='mfi_overbought',
                min_val=70,
                max_val=90,
                default=80,
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
            'stoch_rsi_oversold': ParameterSpec(
                name='stoch_rsi_oversold',
                min_val=0.1,
                max_val=0.4,
                default=0.2,
                param_type='float',
                step=0.1,
            ),
            'stoch_rsi_overbought': ParameterSpec(
                name='stoch_rsi_overbought',
                min_val=0.6,
                max_val=0.9,
                default=0.8,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.0001,
                max_val=0.005,
                default=0.0005,
                param_type='float',
                step=0.1,
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
        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays with NaN handling
        mfi = np.nan_to_num(indicators['mfi'])
        srsi = indicators['stoch_rsi']
        indicators['stoch_rsi']['k'] = np.nan_to_num(srsi["k"])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (
            (mfi < params["mfi_oversold"])
            & (indicators['stoch_rsi']['k'] < params["stoch_rsi_oversold"])
            & (atr > params["atr_min"])
        )
        short_mask = (
            (mfi > params["mfi_overbought"])
            & (indicators['stoch_rsi']['k'] > params["stoch_rsi_overbought"])
            & (atr > params["atr_min"])
        )

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute and write SL/TP for long entries
        if long_mask.any():
            entry_price_long = close[long_mask]
            df.loc[long_mask, "bb_stop_long"] = entry_price_long - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = entry_price_long + params["tp_atr_mult"] * atr[long_mask]

        # Compute and write SL/TP for short entries
        if short_mask.any():
            entry_price_short = close[short_mask]
            df.loc[short_mask, "bb_stop_short"] = entry_price_short + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = entry_price_short - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
