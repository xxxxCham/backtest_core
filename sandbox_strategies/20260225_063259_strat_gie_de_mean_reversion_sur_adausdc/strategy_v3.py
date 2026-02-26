from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mean_reversion_cci_obv_atr')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'obv', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_min': 0.0005,
         'atr_period': 14,
         'cci_extreme': 200,
         'cci_neutral': 50,
         'cci_period': 20,
         'leverage': 1,
         'obv_lookback': 1,
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
            'cci_neutral': ParameterSpec(
                name='cci_neutral',
                min_val=20,
                max_val=100,
                default=50,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=7,
                max_val=28,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_min': ParameterSpec(
                name='atr_min',
                min_val=0.0001,
                max_val=0.005,
                default=0.0005,
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
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
                max_val=200,
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
        # Initialize signals and masks
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract and clean indicators
        cci = np.nan_to_num(indicators['cci'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])

        # OBV reversal detection
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan
        obv_up = obv > prev_obv
        obv_down = obv < prev_obv

        # ATR volatility filter
        atr_min = params.get("atr_min", 0.0005)
        atr_ok = atr >= atr_min

        # CCI thresholds
        cci_extreme = params.get("cci_extreme", 200)
        cci_neutral = params.get("cci_neutral", 50)

        # Entry conditions
        long_entry = (cci <= -cci_extreme) & obv_up & atr_ok
        short_entry = (cci >= cci_extreme) & obv_down & atr_ok

        # Apply entry masks
        long_mask = long_entry
        short_mask = short_entry
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit condition: CCI returns to neutral zone
        exit_mask = np.abs(cci) < cci_neutral
        signals[exit_mask] = 0.0

        # Prepare SL/TP columns
        df["bb_stop_long"] = np.nan
        df["bb_tp_long"] = np.nan
        df["bb_stop_short"] = np.nan
        df["bb_tp_short"] = np.nan

        # ATR‑based stop‑loss and take‑profit levels
        close = df["close"].values
        stop_mult = params.get("stop_atr_mult", 1.1)
        tp_mult = params.get("tp_atr_mult", 3.2)

        # Long SL/TP
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_mult * atr[long_mask]

        # Short SL/TP
        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
