from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='matic_usdc_williams_r_mean_reversion')

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
            'warmup': 30,
            'williams_r_overbought': -20,
            'williams_r_oversold': -80
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'williams_r_oversold': ParameterSpec(
                name='williams_r_oversold',
                min_val=-95,
                max_val=-70,
                default=-80,
                param_type='int',
                step=1,
            ),
            'williams_r_overbought': ParameterSpec(
                name='williams_r_overbought',
                min_val=-30,
                max_val=-10,
                default=-20,
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
            'stoch_rsi_overbought': ParameterSpec(
                name='stoch_rsi_overbought',
                min_val=70,
                max_val=90,
                default=80,
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
                min_val=1.5,
                max_val=4.0,
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
        warmup = int(params.get('warmup', 50))
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicator arrays
        williams = np.nan_to_num(indicators['williams_r'])
        # stoch_rsi is a dict; use its 'k' component
        stoch_k = np.nan_to_num(indicators['stoch_rsi']['k'])
        atr_arr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (
            (williams <= params["williams_r_oversold"])
            & (stoch_k <= params["stoch_rsi_oversold"])
        )
        short_mask = (
            (williams >= params["williams_r_overbought"])
            & (stoch_k >= params["stoch_rsi_overbought"])
        )

        # Apply entry signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long_mask = (williams > -50) & (signals == 1.0)
        exit_short_mask = (williams < -50) & (signals == -1.0)

        signals[exit_long_mask] = 0.0
        signals[exit_short_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Calculate SL/TP on entry bars
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr_arr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr_arr[long_mask]
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr_arr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr_arr[short_mask]

        return signals