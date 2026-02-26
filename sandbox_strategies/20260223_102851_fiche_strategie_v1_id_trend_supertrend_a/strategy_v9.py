from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='trend_supertrend_sma_filter')

    @property
    def required_indicators(self) -> List[str]:
        return ['supertrend', 'adx', 'sma', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1, 'stop_atr_mult': 2.0, 'tp_atr_mult': 3.0, 'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.75,
                param_type='float',
                step=0.1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=1.0,
                max_val=5.0,
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
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)

        # wrap indicator arrays
        close = df["close"].values
        atr = np.nan_to_num(indicators['atr'])
        sma = np.nan_to_num(indicators['sma'])
        st_dir = np.nan_to_num(indicators['supertrend']["direction"])
        adx_val = np.nan_to_num(indicators['adx']["adx"])

        # Entry conditions
        long_mask = (st_dir == 1.0) & (adx_val > 25.0) & (close > sma)
        short_mask = (st_dir == -1.0) & (adx_val > 25.0) & (close < sma)

        # Avoid consecutive duplicate signals
        # Shift signals to compare with previous
        prev_signals = np.roll(signals.values, 1)
        prev_signals[0] = 0.0
        # Only trigger if previous signal was not same
        long_mask = long_mask & (prev_signals != 1.0)
        short_mask = short_mask & (prev_signals != -1.0)

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Exit conditions: change in direction or weak ADX
        exit_long = (st_dir == -1.0) | (adx_val < 20.0)
        exit_short = (st_dir == 1.0) | (adx_val < 20.0)

        # If already flat, no need to set; we keep 0.0 by default
        # But we could set signals to 0 on exit bars to explicitly flat
        # Here we rely on engine to close positions on signal change to 0

        # Warmup protection
        signals.iloc[:warmup] = 0.0

        # Write ATR‑based SL/TP columns
        df.loc[:, "bb_stop_long"] = np.nan
        df.loc[:, "bb_tp_long"] = np.nan
        df.loc[:, "bb_stop_short"] = np.nan
        df.loc[:, "bb_tp_short"] = np.nan

        stop_atr_mult = float(params.get("stop_atr_mult", 1.75))
        tp_atr_mult = float(params.get("tp_atr_mult", 2.5))

        # Long entry levels
        entry_long = signals == 1.0
        df.loc[entry_long, "bb_stop_long"] = close[entry_long] - stop_atr_mult * atr[entry_long]
        df.loc[entry_long, "bb_tp_long"] = close[entry_long] + tp_atr_mult * atr[entry_long]

        # Short entry levels
        entry_short = signals == -1.0
        df.loc[entry_short, "bb_stop_short"] = close[entry_short] + stop_atr_mult * atr[entry_short]
        df.loc[entry_short, "bb_tp_short"] = close[entry_short] - tp_atr_mult * atr[entry_short]
        signals.iloc[:warmup] = 0.0
        return signals
