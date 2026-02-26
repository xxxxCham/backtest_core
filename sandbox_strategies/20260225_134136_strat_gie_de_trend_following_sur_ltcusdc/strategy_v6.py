from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='ltcusdc_1w_sma_aroon_trend_v3')

    @property
    def required_indicators(self) -> List[str]:
        return ['sma', 'aroon', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'aroon_period': 14,
         'atr_period': 14,
         'leverage': 1,
         'sma_period': 50,
         'stop_atr_mult': 2.2,
         'tp_atr_mult': 6.6,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'sma_period': ParameterSpec(
                name='sma_period',
                min_val=10,
                max_val=200,
                default=50,
                param_type='int',
                step=1,
            ),
            'aroon_period': ParameterSpec(
                name='aroon_period',
                min_val=5,
                max_val=30,
                default=14,
                param_type='int',
                step=1,
            ),
            'atr_period': ParameterSpec(
                name='atr_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=2.2,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=10.0,
                default=6.6,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Wrap indicators
        close = df["close"].values
        sma = np.nan_to_num(indicators['sma'])
        indicators['aroon']['aroon_up'] = np.nan_to_num(indicators['aroon']["aroon_up"])
        indicators['aroon']['aroon_down'] = np.nan_to_num(indicators['aroon']["aroon_down"])
        atr = np.nan_to_num(indicators['atr'])

        # Cross helpers
        prev_close = np.roll(close, 1)
        prev_sma = np.roll(sma, 1)
        prev_close[0] = np.nan
        prev_sma[0] = np.nan
        cross_up = (close > sma) & (prev_close <= prev_sma)
        cross_down = (close < sma) & (prev_close >= prev_sma)

        # Long entry conditions
        long_mask = cross_up & (indicators['aroon']['aroon_up'] > 70.0) & (indicators['aroon']['aroon_down'] < 30.0)

        # Short entry conditions
        short_mask = cross_down & (indicators['aroon']['aroon_down'] > 70.0) & (indicators['aroon']['aroon_up'] < 30.0)

        # Exit conditions
        exit_long = (close < sma) | (indicators['aroon']['aroon_up'] < 30.0) | (indicators['aroon']['aroon_down'] > 70.0)
        exit_short = (close > sma) | (indicators['aroon']['aroon_up'] < 30.0) | (indicators['aroon']['aroon_down'] > 70.0)

        # Apply signals
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Exit signals set to 0 (flat)
        signals[exit_long | exit_short] = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # ATR-based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 2.2))
        tp_atr_mult = float(params.get("tp_atr_mult", 6.6))

        entry_long_mask = signals == 1.0
        entry_short_mask = signals == -1.0

        df.loc[entry_long_mask, "bb_stop_long"] = close[entry_long_mask] - stop_atr_mult * atr[entry_long_mask]
        df.loc[entry_long_mask, "bb_tp_long"] = close[entry_long_mask] + tp_atr_mult * atr[entry_long_mask]

        df.loc[entry_short_mask, "bb_stop_short"] = close[entry_short_mask] + stop_atr_mult * atr[entry_short_mask]
        df.loc[entry_short_mask, "bb_tp_short"] = close[entry_short_mask] - tp_atr_mult * atr[entry_short_mask]
        signals.iloc[:warmup] = 0.0
        return signals
