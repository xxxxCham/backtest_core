from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='suiusdc_supertrend_rsi_atr_30m')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'rsi', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'leverage': 1,
         'rsi_period': 14,
         'stop_atr_mult': 1.0,
         'supertrend_multiplier': 3.0,
         'supertrend_period': 10,
         'tp_atr_mult': 1.7,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=14,
                param_type='int',
                step=1,
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
                default=1.0,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=3.0,
                default=1.7,
                param_type='float',
                step=0.1,
            ),
            'warmup': ParameterSpec(
                name='warmup',
                min_val=20,
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
        # Boolean masks for long/short entries and exits
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Extract indicator arrays
        close = df["close"].values
        supertrend_level = np.nan_to_num(indicators['supertrend']["supertrend"])
        rsi_arr = np.nan_to_num(indicators['rsi'])
        atr_arr = np.nan_to_num(indicators['atr'])

        # Entry conditions
        entry_long = (close > supertrend_level) & (rsi_arr > 55.0)
        entry_short = (close < supertrend_level) & (rsi_arr < 45.0)

        long_mask[entry_long] = True
        short_mask[entry_short] = True

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions
        exit_long = (close < supertrend_level) | (rsi_arr < 50.0)
        exit_short = (close > supertrend_level) | (rsi_arr > 50.0)

        # Apply exits by setting to 0 on bars where conditions met
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        # ATR-based stop‑loss and take‑profit levels
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        # Long entry SL/TP
        long_entries = signals == 1.0
        df.loc[long_entries, "bb_stop_long"] = close[long_entries] - params["stop_atr_mult"] * atr_arr[long_entries]
        df.loc[long_entries, "bb_tp_long"] = close[long_entries] + params["tp_atr_mult"] * atr_arr[long_entries]

        # Short entry SL/TP
        short_entries = signals == -1.0
        df.loc[short_entries, "bb_stop_short"] = close[short_entries] + params["stop_atr_mult"] * atr_arr[short_entries]
        df.loc[short_entries, "bb_tp_short"] = close[short_entries] - params["tp_atr_mult"] * atr_arr[short_entries]
        signals.iloc[:warmup] = 0.0
        return signals
