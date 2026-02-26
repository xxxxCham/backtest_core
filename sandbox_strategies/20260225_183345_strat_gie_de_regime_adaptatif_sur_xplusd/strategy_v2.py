from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='atr_adx_obv_regime_adaptive')

    @property
    def required_indicators(self) -> List[str]:
        return ['atr', 'adx', 'obv']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'atr_threshold_mult': 1.5,
         'leverage': 1,
         'stop_atr_mult': 1.4,
         'tp_atr_mult': 3.36,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_threshold_mult': ParameterSpec(
                name='atr_threshold_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=3.36,
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
        # Boolean masks for long/short
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        signals.iloc[:warmup] = 0.0

        # Wrap indicators
        atr = np.nan_to_num(indicators['atr'])
        obv = np.nan_to_num(indicators['obv'])

        # High/low 20-bar extremes
        high20 = df["high"].rolling(20).max().values
        low20 = df["low"].rolling(20).min().values

        # OBV slope
        obv_prev = np.roll(obv, 1)
        obv_prev[0] = np.nan
        obv_up = obv > obv_prev
        obv_down = obv < obv_prev

        # Volatility regime
        atr_mean = atr.mean()
        atr_threshold_val = params["atr_threshold_mult"] * atr_mean
        high_vol = atr > atr_threshold_val
        low_vol = ~high_vol

        close = df["close"].values

        # Long entry conditions
        long_cond1 = high_vol & (close > high20) & obv_up
        long_cond2 = low_vol & (close < low20) & obv_up
        long_mask = long_cond1 | long_cond2

        # Short entry conditions
        short_cond1 = high_vol & (close < low20) & obv_down
        short_cond2 = low_vol & (close > high20) & obv_down
        short_mask = short_cond1 | short_cond2

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Risk management: write SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        entry_long = long_mask
        entry_short = short_mask

        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
