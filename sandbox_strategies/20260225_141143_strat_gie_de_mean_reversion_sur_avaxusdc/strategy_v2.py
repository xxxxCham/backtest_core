from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='keltner_williams_cci_mean_reversion')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'williams_r', 'cci', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'cci_overbought': 100,
         'cci_oversold': -100,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 2.2,
         'warmup': 30,
         'williams_r_overbought': -20,
         'williams_r_oversold': -80}

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
            'cci_overbought': ParameterSpec(
                name='cci_overbought',
                min_val=80,
                max_val=120,
                default=100,
                param_type='int',
                step=5,
            ),
            'cci_oversold': ParameterSpec(
                name='cci_oversold',
                min_val=-120,
                max_val=-80,
                default=-100,
                param_type='int',
                step=5,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.0,
                default=1.1,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=4.0,
                default=2.2,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        kelt = indicators['keltner']
        lower = np.nan_to_num(kelt["lower"])
        middle = np.nan_to_num(kelt["middle"])
        upper = np.nan_to_num(kelt["upper"])
        williams = np.nan_to_num(indicators['williams_r'])
        cci = np.nan_to_num(indicators['cci'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Entry conditions
        long_mask = (close < lower) & (williams <= params["williams_r_oversold"]) & (cci <= params["cci_oversold"])
        short_mask = (close > upper) & (williams >= params["williams_r_overbought"]) & (cci >= params["cci_overbought"])

        # Exit conditions: cross of close with middle or cci with 0
        prev_close = np.roll(close, 1)
        prev_middle = np.roll(middle, 1)
        prev_cci = np.roll(cci, 1)
        prev_close[0] = np.nan
        prev_middle[0] = np.nan
        prev_cci[0] = np.nan

        cross_close_middle = ((close > middle) & (prev_close <= prev_middle)) | ((close < middle) & (prev_close >= prev_middle))
        cross_cci_0 = ((cci > 0) & (prev_cci <= 0)) | ((cci < 0) & (prev_cci >= 0))
        exit_mask = cross_close_middle | cross_cci_0

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
