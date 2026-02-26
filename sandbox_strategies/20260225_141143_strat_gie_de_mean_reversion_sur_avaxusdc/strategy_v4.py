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
        return {'leverage': 1, 'stop_atr_mult': 1.1, 'tp_atr_mult': 2.0, 'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
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
                max_val=4.0,
                default=2.0,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=50,
                default=30,
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
        n = len(df)
        warmup = int(params.get('warmup', 50))

        # Initialize signal series
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Pull indicator arrays
        close = df["close"].values
        kelt = indicators['keltner']
        lower = np.nan_to_num(kelt["lower"])
        upper = np.nan_to_num(kelt["upper"])
        middle = np.nan_to_num(kelt["middle"])
        williams = np.nan_to_num(indicators['williams_r'])
        cci_arr = np.nan_to_num(indicators['cci'])
        atr_arr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        long_mask = (close < lower) & (williams <= -80) & (cci_arr <= -100)
        short_mask = (close > upper) & (williams >= -20) & (cci_arr >= 100)

        # Exit conditions: cross of close with middle band or cci crossing 0
        close_prev = np.roll(close, 1)
        middle_prev = np.roll(middle, 1)
        close_prev[0] = np.nan
        middle_prev[0] = np.nan
        close_cross = ((close > middle) & (close_prev <= middle_prev)) | ((close < middle) & (close_prev >= middle_prev))

        cci_prev = np.roll(cci_arr, 1)
        cci_prev[0] = np.nan
        cci_cross = ((cci_arr > 0) & (cci_prev <= 0)) | ((cci_arr < 0) & (cci_prev >= 0))

        exit_mask = close_cross | cci_cross

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0

        # Warmup period
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.1))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr_arr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr_arr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr_arr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr_arr[short_mask]

        return signals