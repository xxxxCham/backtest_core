from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mean_reversion_cci_keltner')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'cci_extreme': 200,
         'cci_period': 20,
         'keltner_multiplier': 1.5,
         'keltner_period': 20,
         'leverage': 1,
         'stop_atr_mult': 1.1,
         'tp_atr_mult': 3.2,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'cci_period': ParameterSpec(
                name='cci_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'cci_extreme': ParameterSpec(
                name='cci_extreme',
                min_val=100,
                max_val=300,
                default=200,
                param_type='int',
                step=1,
            ),
            'keltner_period': ParameterSpec(
                name='keltner_period',
                min_val=10,
                max_val=40,
                default=20,
                param_type='int',
                step=1,
            ),
            'keltner_multiplier': ParameterSpec(
                name='keltner_multiplier',
                min_val=1.0,
                max_val=3.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
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
                max_val=5.0,
                default=3.2,
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
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)

        # Initialize masks
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Extract price and indicators, ensure NaNs are handled
        close = df["close"].values
        cci = np.nan_to_num(indicators['cci'])
        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])
        atr = np.nan_to_num(indicators['atr'])

        # Parameters
        cci_extreme = params.get("cci_extreme", 200)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 3.2)
        warmup = int(params.get("warmup", 50))

        # Entry conditions
        long_entry = (close < indicators['keltner']['lower']) & (cci <= -cci_extreme)
        short_entry = (close > indicators['keltner']['upper']) & (cci >= cci_extreme)

        # Exit condition (neutral CCI)
        exit_cond = np.abs(cci) < 50

        # Apply masks
        long_mask[long_entry] = True
        short_mask[short_entry] = True

        # Set signals
        signals[exit_cond] = 0.0
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Prepare SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Write ATR‑based stop‑loss and take‑profit for long entries
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        # Write ATR‑based stop‑loss and take‑profit for short entries
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
