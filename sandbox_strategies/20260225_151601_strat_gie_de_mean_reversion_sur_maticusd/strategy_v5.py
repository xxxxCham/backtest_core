from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='maticusdc_mean_reversion_williams_stoch_rsi')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'stoch_rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'stoch_rsi_overbought': 80,
            'stoch_rsi_oversold': 20,
            'stop_atr_mult': 1.2,
            'tp_atr_mult': 2.9,
            'warmup': 50,
            'williams_overbought': -20,
            'williams_oversold': -80,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_period': ParameterSpec(
                name='williams_period',
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
                max_val=4.0,
                default=1.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.9,
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        warmup = int(params.get('warmup', 50))
        # Ensure signals are zero during warm‑up
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays safely
        williams = np.nan_to_num(indicators['williams_r'])
        stoch_k = np.nan_to_num(indicators['stoch_rsi']['k'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (
            (williams <= params["williams_oversold"])
            & (stoch_k <= params["stoch_rsi_oversold"])
        )
        short_mask = (
            (williams >= params["williams_overbought"])
            & (stoch_k >= params["stoch_rsi_overbought"])
        )

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Compute ATR‑based levels on entry bars
        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - params["stop_atr_mult"] * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + params["tp_atr_mult"] * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + params["stop_atr_mult"] * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - params["tp_atr_mult"] * atr[entry_short]

        # Re‑apply warm‑up protection to avoid signals on early bars
        signals.iloc[:warmup] = 0.0
        return signals