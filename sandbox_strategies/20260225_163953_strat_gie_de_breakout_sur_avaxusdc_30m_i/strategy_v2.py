from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='avax_bollinger_atr_trend_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'adx']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_min': 0.5,
         'leverage': 1,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 4.8,
         'trailing_atr_mult': 2.3,
         'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.1,
                max_val=5.0,
                default=0.5,
                param_type='float',
                step=0.1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=4.8,
                param_type='float',
                step=0.1,
            ),
            'trailing_atr_mult': ParameterSpec(
                name='trailing_atr_mult',
                min_val=1.0,
                max_val=5.0,
                default=2.3,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=10,
                max_val=100,
                default=20,
                param_type='int',
                step=10,
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
        # Boolean masks initialization
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract indicator arrays
        close = df["close"].values
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators['atr'])
        adx_d = indicators['adx']
        adx_val = np.nan_to_num(adx_d["adx"])

        # Helper cross functions
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        cross_long = (close > upper) & (prev_close <= upper)
        cross_short = (close < lower) & (prev_close >= lower)

        # Entry conditions
        long_mask = (
            cross_long
            & (adx_val > 25)
            & (atr > params.get("atr_min", 0.5))
        )
        short_mask = (
            cross_short
            & (adx_val > 25)
            & (atr > params.get("atr_min", 0.5))
        )

        # Exit conditions
        prev_middle = np.roll(middle, 1)
        prev_middle[0] = np.nan
        cross_down_middle = (close < middle) & (prev_close >= prev_middle)
        cross_up_middle = (close > middle) & (prev_close <= prev_middle)

        exit_long = cross_down_middle | (adx_val < 20)
        exit_short = cross_up_middle | (adx_val < 20)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = (
            close[entry_long] - params.get("stop_atr_mult", 1.5) * atr[entry_long]
        )
        df.loc[entry_long, "bb_tp_long"] = (
            close[entry_long] + params.get("tp_atr_mult", 4.8) * atr[entry_long]
        )
        df.loc[entry_short, "bb_stop_short"] = (
            close[entry_short] + params.get("stop_atr_mult", 1.5) * atr[entry_short]
        )
        df.loc[entry_short, "bb_tp_short"] = (
            close[entry_short] - params.get("tp_atr_mult", 4.8) * atr[entry_short]
        )
        signals.iloc[:warmup] = 0.0
        return signals
