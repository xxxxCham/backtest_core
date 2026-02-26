from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='stoch_rsi_obv_bollinger_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['stoch_rsi', 'obv', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            'leverage': 1,
            'stoch_rsi_overbought': 80,
            'stoch_rsi_oversold': 20,
            'stoch_rsi_period': 14,
            'stop_atr_mult': 2.3,
            'tp_atr_mult': 4.5,
            'warmup': 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stoch_rsi_period': ParameterSpec(
                name='stoch_rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stoch_rsi_overbought': ParameterSpec(
                name='stoch_rsi_overbought',
                min_val=60,
                max_val=90,
                default=80,
                param_type='int',
                step=1,
            ),
            'stoch_rsi_oversold': ParameterSpec(
                name='stoch_rsi_oversold',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=4.5,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Extract indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        stoch_k = np.nan_to_num(indicators['stoch_rsi']["k"])
        obv = np.nan_to_num(indicators['obv'])
        boll = indicators['bollinger']
        lower = np.nan_to_num(boll["lower"])
        middle = np.nan_to_num(boll["middle"])
        upper = np.nan_to_num(boll["upper"])

        # Previous OBV for trend confirmation
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan

        # Entry conditions
        oversold = params.get("stoch_rsi_oversold", 20)
        overbought = params.get("stoch_rsi_overbought", 80)

        long_mask = (stoch_k < oversold) & (obv > obv_prev) & (close < lower)
        short_mask = (stoch_k > overbought) & (obv < obv_prev) & (close > upper)

        # Exit conditions: cross of close with middle or stoch_k with 50
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan
        cross_close_mid = ((close > middle) & (prev_close <= prev_middle)) | ((close < middle) & (prev_close >= prev_middle))

        prev_k = np.roll(stoch_k, 1)
        prev_k[0] = np.nan
        cross_k_50 = ((stoch_k > 50) & (prev_k <= 50)) | ((stoch_k < 50) & (prev_k >= 50))

        exit_mask = cross_close_mid | cross_k_50

        # Apply warmup
        valid = np.arange(n) >= warmup
        long_mask &= valid
        short_mask &= valid
        exit_mask &= valid

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # ATR-based SL/TP for long entries
        stop_mult = params.get("stop_atr_mult", 2.3)
        tp_mult = params.get("tp_atr_mult", 4.5)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # ATR-based SL/TP for short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        signals.iloc[:warmup] = 0.0
        return signals