from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='bollinger_atr_sma_breakout')

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'atr', 'sma']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'sma_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 4.8,
         'trailing_atr_mult': 2.3,
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
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        # Boolean masks for long and short entries
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        bb = indicators['bollinger']
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators['atr'])
        sma = np.nan_to_num(indicators['sma'])
        close = df["close"].values

        # Entry conditions
        band_width = upper - lower
        atr_mult = params.get("stop_atr_mult", 1.5)
        tp_mult = params.get("tp_atr_mult", 4.8)

        long_cond = (
            (close > upper)
            & (close > sma)
            & (band_width > 1.5 * atr)
        )
        short_cond = (
            (close < lower)
            & (close < sma)
            & (band_width > 1.5 * atr)
        )

        long_mask[long_cond] = True
        short_mask[short_cond] = True

        # Avoid duplicate consecutive signals
        dup_long = (signals == 1.0) & (np.roll(signals, 1) == 1.0)
        dup_short = (signals == -1.0) & (np.roll(signals, 1) == -1.0)
        long_mask[dup_long] = False
        short_mask[dup_short] = False

        # Assign signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Set stop‑loss and take‑profit columns for long positions
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        entry_long = signals == 1.0
        entry_short = signals == -1.0

        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_mult * atr[entry_long]

        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_mult * atr[entry_short]

        # Exit logic: cross middle band or trend reversal relative to SMA
        # We use a simple flat signal for exits; simulator handles position closing on 0
        # No explicit exit signals are set in this series
        signals.iloc[:warmup] = 0.0
        return signals
