from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='mean_reversion_cci_bollinger_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'cci_overshoot': 100,
         'cci_period': 20,
         'leverage': 1,
         'stop_atr_mult': 2.4,
         'tp_atr_mult': 2.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_period': ParameterSpec(
                name='cci_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4,
                default=2.4,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1,
                max_val=5,
                default=2.5,
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
        # Prepare indicator arrays
        close = df["close"].values
        boll = indicators['bollinger']
        upper = np.nan_to_num(boll["upper"])
        middle = np.nan_to_num(boll["middle"])
        lower = np.nan_to_num(boll["lower"])
        cci = np.nan_to_num(indicators['cci'])
        atr = np.nan_to_num(indicators['atr'])

        # Masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Entry conditions
        cci_overshoot = params.get("cci_overshoot", 100)
        long_mask = (close < lower) & (cci < -cci_overshoot)
        short_mask = (close > upper) & (cci > cci_overshoot)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: cross of close with middle band
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_up = (close > middle) & (prev_close <= middle)
        cross_down = (close < middle) & (prev_close >= middle)

        long_exit_mask = (signals == 1.0) & cross_up
        short_exit_mask = (signals == -1.0) & cross_down

        signals[long_exit_mask] = 0.0
        signals[short_exit_mask] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # ATR-based SL/TP on entry bars
        stop_atr_mult = params.get("stop_atr_mult", 2.4)
        tp_atr_mult = params.get("tp_atr_mult", 2.5)

        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
