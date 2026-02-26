from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='vwap_stochastic_ema_scalp')

    @property
    def required_indicators(self) -> List[str]:
        return ['vwap', 'stochastic', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_period': 20,
         'leverage': 1,
         'stochastic_k_period': 14,
         'stop_atr_mult': 2.1,
         'tp_atr_mult': 2.9,
         'vwap_period': 20,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'vwap_period': ParameterSpec(
                name='vwap_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stochastic_k_period': ParameterSpec(
                name='stochastic_k_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'ema_period': ParameterSpec(
                name='ema_period',
                min_val=5,
                max_val=50,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.1,
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

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        warmup = int(params.get('warmup', 50))
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Indicator arrays
        close = df["close"].values
        vwap = np.nan_to_num(indicators['vwap'])
        stochastic_k = np.nan_to_num(indicators['stochastic']["stoch_k"])
        ema = np.nan_to_num(indicators['ema'])
        atr = np.nan_to_num(indicators['atr'])

        # Helper cross functions
        prev_close = np.roll(close, 1)
        prev_vwap = np.roll(vwap, 1)
        prev_close[0] = np.nan
        prev_vwap[0] = np.nan
        cross_up = (close > vwap) & (prev_close <= prev_vwap)
        cross_down = (close < vwap) & (prev_close >= prev_vwap)

        # Long entry: close crosses above vwap AND stochastic_k > 80 AND close > ema
        long_mask = cross_up & (stochastic_k > 80) & (close > ema)

        # Short entry: close crosses below vwap AND stochastic_k < 20 AND close < ema
        short_mask = cross_down & (stochastic_k < 20) & (close < ema)

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit logic: reverse cross of vwap
        exit_long_mask = cross_down & (signals == 1.0)
        exit_short_mask = cross_up & (signals == -1.0)

        # Set exit to flat (0.0)
        signals[exit_long_mask | exit_short_mask] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        stop_atr_mult = float(params.get("stop_atr_mult", 2.1))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.9))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
