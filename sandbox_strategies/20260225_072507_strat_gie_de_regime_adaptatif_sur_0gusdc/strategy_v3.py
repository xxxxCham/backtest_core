from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adaptive_regime_keltner_supertrend_bollinger')

    @property
    def required_indicators(self) -> List[str]:
        return ['keltner', 'supertrend', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'keltner_mult': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.6,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 3.5,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_mult': ParameterSpec(
                name='keltner_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'supertrend_period': ParameterSpec(
                name='supertrend_period',
                min_val=5,
                max_val=30,
                default=10,
                param_type='int',
                step=1,
            ),
            'supertrend_multiplier': ParameterSpec(
                name='supertrend_multiplier',
                min_val=1.0,
                max_val=5.0,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.6,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=6.0,
                default=3.5,
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
        # Extract price series
        close = df["close"].values

        # Extract and sanitize indicators
        kelt = indicators['keltner']
        kelt_mid = np.nan_to_num(kelt["middle"])

        st = indicators['supertrend']
        st_dir = np.nan_to_num(st["direction"])

        bb = indicators['bollinger']
        indicators['bollinger']['upper'] = np.nan_to_num(bb["upper"])
        indicators['bollinger']['lower'] = np.nan_to_num(bb["lower"])
        indicators['bollinger']['middle'] = np.nan_to_num(bb["middle"])

        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        stop_atr_mult = float(params.get("stop_atr_mult", 1.6))
        tp_atr_mult = float(params.get("tp_atr_mult", 3.5))
        warmup = int(params.get("warmup", 50))

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        long_entry = (close > kelt_mid) & (st_dir > 0) & (close > indicators['bollinger']['upper'])
        short_entry = (close < kelt_mid) & (st_dir < 0) & (close < indicators['bollinger']['lower'])

        # Populate masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Set signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Initialize SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # Compute SL/TP on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        # Exit conditions (flatten signals on exit bars – signals already zero by default)
        exit_long = (close < indicators['bollinger']['middle']) | (st_dir <= 0)
        exit_short = (close > indicators['bollinger']['middle']) | (st_dir >= 0)

        # Ensure no lingering entry signals when exit condition met
        signals[exit_long & long_mask] = 0.0
        signals[exit_short & short_mask] = 0.0

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
