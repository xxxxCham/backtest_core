from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='adausdc_mean_reversion_cci_obv_keltner')

    @property
    def required_indicators(self) -> List[str]:
        return ['cci', 'obv', 'keltner', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'cci_overbought': 100,
         'cci_oversold': -100,
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
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=2.5,
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
        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays and wrap with nan_to_num
        close = df["close"].values
        cci = np.nan_to_num(indicators['cci'])
        obv = np.nan_to_num(indicators['obv'])
        atr = np.nan_to_num(indicators['atr'])

        kelt = indicators['keltner']
        indicators['keltner']['upper'] = np.nan_to_num(kelt["upper"])
        indicators['keltner']['lower'] = np.nan_to_num(kelt["lower"])

        # Previous OBV for reversal detection
        prev_obv = np.roll(obv, 1)
        prev_obv[0] = np.nan

        # Parameter shortcuts
        cci_overbought = params.get("cci_overbought", 100)
        cci_oversold = params.get("cci_oversold", -100)
        stop_atr_mult = params.get("stop_atr_mult", 1.1)
        tp_atr_mult = params.get("tp_atr_mult", 3.2)

        # Long entry: price below lower Keltner, CCI oversold, OBV rising
        long_mask = (close < indicators['keltner']['lower']) & (cci <= cci_oversold) & (obv > prev_obv)

        # Short entry: price above upper Keltner, CCI overbought, OBV falling
        short_mask = (close > indicators['keltner']['upper']) & (cci >= cci_overbought) & (obv < prev_obv)

        # Exit condition: CCI near neutral (optional flat signal)
        exit_mask = np.abs(cci) < 50

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals[exit_mask] = 0.0  # ensure flat when exit condition met

        # Initialize SL/TP columns with NaN
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Set ATR‑based stop‑loss and take‑profit on entry bars
        df.loc[long_mask, "bb_stop_long"] = close[long_mask] - stop_atr_mult * atr[long_mask]
        df.loc[long_mask, "bb_tp_long"] = close[long_mask] + tp_atr_mult * atr[long_mask]

        df.loc[short_mask, "bb_stop_short"] = close[short_mask] + stop_atr_mult * atr[short_mask]
        df.loc[short_mask, "bb_tp_short"] = close[short_mask] - tp_atr_mult * atr[short_mask]

        return signals
        signals.iloc[:warmup] = 0.0
        return signals
