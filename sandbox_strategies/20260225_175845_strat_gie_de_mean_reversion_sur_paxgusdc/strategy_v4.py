from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='paxg_williams_obv_atr_mean_reversion_v2')

    @property
    def required_indicators(self) -> List[str]:
        return ['williams_r', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'stop_atr_mult': 1.7,
         'tp_atr_mult': 2.4,
         'warmup': 30,
         'williams_period': 14}

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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=2.4,
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
        # Extract indicator arrays
        williams = np.nan_to_num(indicators['williams_r'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])
        close = df["close"].values

        # Previous OBV values for trend confirmation
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan
        obv_prev2 = np.roll(obv, 2)
        obv_prev2[0] = np.nan
        obv_prev2[1] = np.nan

        # Long entry: Williams %R below -80 and OBV trending up
        long_mask = (williams < -80) & (obv > obv_prev) & (obv_prev > obv_prev2)

        # Short entry: Williams %R above -20 and OBV trending down
        short_mask = (williams > -20) & (obv < obv_prev) & (obv_prev < obv_prev2)

        # Cross detection for exit at -50
        prev_w = np.roll(williams, 1)
        prev_w[0] = np.nan
        cross_up = (williams > -50) & (prev_w <= -50)
        cross_down = (williams < -50) & (prev_w >= -50)
        exit_mask = cross_up | cross_down

        # Build temporary signal series for forward filling
        temp = pd.Series(0.0, index=df.index, dtype=np.float64)
        temp[long_mask] = 1.0
        temp[short_mask] = -1.0
        temp[exit_mask] = 0.0
        # Forward fill to maintain position until exit

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Initialize SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based stop and take profit for long entries
        if long_mask.any():
            df.loc[long_mask, "bb_stop_long"] = close[long_mask] - params["stop_atr_mult"] * atr[long_mask]
            df.loc[long_mask, "bb_tp_long"] = close[long_mask] + params["tp_atr_mult"] * atr[long_mask]

        # ATR-based stop and take profit for short entries
        if short_mask.any():
            df.loc[short_mask, "bb_stop_short"] = close[short_mask] + params["stop_atr_mult"] * atr[short_mask]
            df.loc[short_mask, "bb_tp_short"] = close[short_mask] - params["tp_atr_mult"] * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
